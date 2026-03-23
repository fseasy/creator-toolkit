from __future__ import annotations

import io
import logging
import threading
import zipfile
from pathlib import Path

import httpx
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fs_pyutils.audio import audio_to_mp3_bytes

from .cache import SpeakerPromptCache
from .models import AudioFormat, BatchTTSRequest, CreateSpeakerResponse, Manifest, ManifestItem, TTSRequest
from .serialize import save_prompt_items, save_ref_audio, save_ref_text
from .tts import TTSEngine

logger = logging.getLogger(__name__)

router = APIRouter()

_db_path: Path | None = None
_tts_engine: TTSEngine | None = None
_speaker_cache: SpeakerPromptCache | None = None


thread_semaphore = threading.Semaphore(2)


def get_db_path() -> Path:
  if _db_path is None:
    raise RuntimeError("Server not initialized")
  return _db_path


def get_tts_engine() -> TTSEngine:
  if _tts_engine is None:
    raise RuntimeError("Server not initialized")
  return _tts_engine


def get_speaker_cache() -> SpeakerPromptCache:
  if _speaker_cache is None:
    raise RuntimeError("Server not initialized")
  return _speaker_cache


def init_app(db_path: Path, model_path: Path) -> None:
  global _db_path, _tts_engine, _speaker_cache
  _db_path = db_path / "speaker"
  _speaker_cache = SpeakerPromptCache(max_size=5)
  _tts_engine = TTSEngine(model_path=model_path)


@router.post("/create-speaker")
def create_speaker(
  speaker_name: str = Form(...),
  ref_audio: UploadFile | None = File(default=None),
  ref_audio_url: str | None = Form(default=None),
  ref_text: str | None = Form(default=None),
  db_path: Path = Depends(get_db_path),
  tts_engine: TTSEngine = Depends(get_tts_engine),
  speaker_cache: SpeakerPromptCache = Depends(get_speaker_cache),
) -> CreateSpeakerResponse:
  if ref_audio is None and ref_audio_url is None:
    raise HTTPException(status_code=400, detail="Either ref_audio file or ref_audio_url is required")

  speaker_db = db_path / speaker_name

  ref_audio_data: bytes

  if ref_audio is not None:
    ref_audio_data = ref_audio.file.read()
  else:
    assert ref_audio_url is not None
    if ref_audio_url.startswith(("http://", "https://")):
      with httpx.Client() as client:
        response = client.get(ref_audio_url)
        response.raise_for_status()
        ref_audio_data = response.content
    elif ref_audio_url.startswith("data:audio/"):
      import base64

      header, data = ref_audio_url.split(",", 1)
      ref_audio_data = base64.b64decode(data)
    else:
      raise HTTPException(status_code=400, detail="Invalid ref_audio_url")

  audio_ext = "wav"
  if ref_audio_data[:4] == b"fLaC":
    audio_ext = "flac"
  elif ref_audio_data[:3] == b"ID3" or ref_audio_data[:10] == b"\x00\x00\x00\x18mpg":
    audio_ext = "mp3"

  audio_name = f"ref_audio.{audio_ext}"
  ref_audio_path = save_ref_audio(ref_audio_data, audio_name, speaker_db)
  save_ref_text(ref_text, speaker_db)

  x_vector_only_mode = ref_text is None or ref_text == ""

  with thread_semaphore:
    prompt_items = tts_engine.create_voice_clone_prompt(
      ref_audio=str(ref_audio_path),
      ref_text=ref_text,
      x_vector_only_mode=x_vector_only_mode,
    )
  save_prompt_items(prompt_items, speaker_db)

  speaker_cache.put(speaker_name, prompt_items)  # cache
  return CreateSpeakerResponse(speaker_name=speaker_name, message=f"Speaker '{speaker_name}' created successfully")


@router.post("/batch-tts")
def batch_tts(
  request: BatchTTSRequest,
  db_path: Path = Depends(get_db_path),
  tts_engine: TTSEngine = Depends(get_tts_engine),
  speaker_cache: SpeakerPromptCache = Depends(get_speaker_cache),
) -> Response:
  if isinstance(request.languages, str):
    languages = [request.languages] * len(request.texts)
  else:
    languages = request.languages
    if len(languages) != len(request.texts):
      raise HTTPException(
        status_code=400,
        detail="Length of languages must match length of texts",
      )

  cached_prompt = speaker_cache.get(request.speaker_name)
  if cached_prompt is not None:
    prompt_items = cached_prompt
  else:
    speaker_db = db_path / request.speaker_name
    if not speaker_db.exists():
      raise HTTPException(
        status_code=404,
        detail=f"Speaker '{request.speaker_name}' not found",
      )
    prompt_items = tts_engine.load_prompt_from_db(speaker_db)
    speaker_cache.put(request.speaker_name, prompt_items)

  with thread_semaphore:
    wavs, sr = tts_engine.generate_voice_clone(
      text=request.texts,
      language=languages,
      voice_clone_prompt=prompt_items,
    )

  manifest_items: list[ManifestItem] = []
  audio_files: list[tuple[str, bytes]] = []

  for i, (wav, text, lang) in enumerate(zip(wavs, request.texts, languages, strict=True)):
    audio_name = f"audio_{i:04d}.wav"
    wav_bytes = _numpy_to_wav_bytes(wav, sr)
    audio_files.append((audio_name, wav_bytes))

    if request.audio_fmt == AudioFormat.MP3:
      mp3_name = f"audio_{i:04d}.mp3"
      mp3_bytes = audio_to_mp3_bytes(wav_bytes)
      audio_files[i] = (mp3_name, mp3_bytes)
      audio_name = mp3_name

    manifest_items.append(ManifestItem(text=text, language=lang, audio_name=audio_name))

  manifest = Manifest(items=manifest_items)
  manifest_bytes = manifest.model_dump_json().encode("utf-8")
  audio_files.append(("manifest.json", manifest_bytes))

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for name, data in audio_files:
      zf.writestr(name, data)

  zip_buffer.seek(0)

  return Response(
    zip_buffer.getvalue(),
    media_type="application/zip",
    headers={"Content-Disposition": "attachment; filename=tts_output.zip"},
  )


@router.post("/tts")
def tts(
  request: TTSRequest,
  db_path: Path = Depends(get_db_path),
  tts_engine: TTSEngine = Depends(get_tts_engine),
  speaker_cache: SpeakerPromptCache = Depends(get_speaker_cache),
) -> Response:
  """! Note: use the sync way"""
  cached_prompt = speaker_cache.get(request.speaker_name)
  if cached_prompt is not None:
    prompt_items = cached_prompt
  else:
    speaker_db = db_path / request.speaker_name
    if not speaker_db.exists():
      raise HTTPException(
        status_code=404,
        detail=f"Speaker '{request.speaker_name}' not found",
      )
    prompt_items = tts_engine.load_prompt_from_db(speaker_db)
    speaker_cache.put(request.speaker_name, prompt_items)

  with thread_semaphore:
    wavs, sr = tts_engine.generate_voice_clone(
      text=[request.text],
      language=[request.language],
      voice_clone_prompt=prompt_items,
    )

  audio_wav = wavs[0]
  audio_bytes = _numpy_to_wav_bytes(audio_wav, sr)
  if request.audio_fmt == AudioFormat.MP3:
    audio_bytes = audio_to_mp3_bytes(audio_bytes)

  return Response(
    audio_bytes,
    media_type=_audio_fmt2media_type(request.audio_fmt),
    headers={
      # Suggests the browser play the file inline
      "Content-Disposition": f"inline; filename=audio.{request.audio_fmt}"
    },
  )


def _numpy_to_wav_bytes(wav: list[float] | list[list[float]], sr: int) -> bytes:
  """Normalize raw PCM and transform it to be .wav format"""
  import numpy as np
  import pyloudnorm as pyln

  wav_array = np.array(wav)
  wav_array = pyln.normalize.peak(wav_array, -1.0)
  if wav_array.ndim > 1:
    wav_array = wav_array.squeeze()
  buffer = io.BytesIO()
  sf.write(buffer, wav_array, sr, format="WAV")
  buffer.seek(0)
  return buffer.getvalue()


def _audio_fmt2media_type(fmt: AudioFormat) -> str:
  media_type = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
  }
  return media_type[fmt]

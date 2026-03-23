from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .models import AudioFormat, BatchTTSRequest, CreateSpeakerResponse, Manifest, TTSRequest


@dataclass
class TTSItem:
  text: str
  language: str
  audio_bytes: bytes


@dataclass
class BatchTTSResult:
  items: list[TTSItem]

  @classmethod
  def from_zip(cls, zip_bytes: bytes) -> BatchTTSResult:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
      manifest = Manifest.model_validate(json.loads(zf.read("manifest.json")))
      items = [
        TTSItem(
          text=item.text,
          language=item.language,
          audio_bytes=zf.read(item.audio_name),
        )
        for item in manifest.items
      ]
    return cls(items=items)


class Qwen3TTSClient:
  def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0) -> None:
    self.base_url = base_url.rstrip("/")
    self.timeout = timeout

  async def create_speaker(
    self,
    speaker_name: str,
    ref_audio: str | Path | bytes,
    ref_text: str | None = None,
  ) -> CreateSpeakerResponse:
    ref_audio_bytes: bytes | None = None
    ref_audio_url: str | None = None

    if isinstance(ref_audio, bytes):
      ref_audio_bytes = ref_audio
    elif isinstance(ref_audio, Path):
      ref_audio_bytes = ref_audio.read_bytes()
    else:
      assert isinstance(ref_audio, str)
      if ref_audio.startswith(("http://", "https://", "data:")):
        ref_audio_url = ref_audio
      else:
        ref_audio_bytes = Path(ref_audio).read_bytes()

    form_data: dict[str, str] = {"speaker_name": speaker_name}
    if ref_text is not None:
      form_data["ref_text"] = ref_text

    files: dict[str, tuple[str, io.BytesIO, str]] | None = None
    if ref_audio_bytes is not None:
      files = {"ref_audio": ("audio.wav", io.BytesIO(ref_audio_bytes), "audio/wav")}
      form_data["ref_audio_url"] = ""
    else:
      form_data["ref_audio_url"] = ref_audio_url or ""
    try:
      async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.post(
          f"{self.base_url}/create-speaker",
          data=form_data,
          files=files,  # type: ignore[arg-type]
        )
      response.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise RuntimeError(
        f"/create-speaker api get exception: code={e.response.status_code}, detail={e.response.text}"
      ) from e
    return CreateSpeakerResponse.model_validate(response.json())

  async def batch_tts(
    self,
    speaker_name: str,
    texts: list[str],
    languages: str | list[str],
    audio_fmt: AudioFormat = AudioFormat.WAV,
  ) -> BatchTTSResult:
    request = BatchTTSRequest(
      speaker_name=speaker_name,
      texts=texts,
      languages=languages,
      audio_fmt=audio_fmt,
    )
    try:
      async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.post(
          f"{self.base_url}/batch-tts",
          json=request.model_dump(),
        )
      response.raise_for_status()
    except httpx.HTTPStatusError as e:
      # let the exception include the response.text! or you can only get the code.
      raise RuntimeError(
        f"/batch-tts api get exception: code={e.response.status_code}, detail={e.response.text}"
      ) from e
    return BatchTTSResult.from_zip(response.content)

  async def tts(
    self,
    speaker_name: str,
    text: str,
    language: str,
    audio_fmt: AudioFormat = AudioFormat.WAV,
  ) -> bytes:
    request = TTSRequest(
      speaker_name=speaker_name,
      text=text,
      language=language,
      audio_fmt=audio_fmt,
    )
    try:
      async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.post(
          f"{self.base_url}/tts",
          json=request.model_dump(),
        )
      response.raise_for_status()
    except httpx.HTTPStatusError as e:
      raise RuntimeError(f"/tts api get exception: code={e.response.status_code}, detail={e.response.text}") from e
    return response.content

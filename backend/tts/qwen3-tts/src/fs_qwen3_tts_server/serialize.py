from __future__ import annotations

import pickle
from pathlib import Path

from qwen_tts import VoiceClonePromptItem


def serialize_prompt_items(items: list[VoiceClonePromptItem]) -> bytes:
  return pickle.dumps(items)


def deserialize_prompt_items(data: bytes) -> list[VoiceClonePromptItem]:
  return pickle.loads(data)


def save_prompt_items(items: list[VoiceClonePromptItem], speaker_db: Path) -> None:
  speaker_db.mkdir(parents=True, exist_ok=True)
  prompt_path = speaker_db / "prompt.pkl"
  prompt_path.write_bytes(serialize_prompt_items(items))


def load_prompt_items(speaker_db: Path) -> list[VoiceClonePromptItem]:
  prompt_path = speaker_db / "prompt.pkl"
  return deserialize_prompt_items(prompt_path.read_bytes())


def save_ref_audio(audio_bytes: bytes, audio_name: str, speaker_db: Path) -> Path:
  speaker_db.mkdir(parents=True, exist_ok=True)
  audio_path = speaker_db / audio_name
  audio_path.write_bytes(audio_bytes)
  return audio_path


def save_ref_text(ref_text: str | None, speaker_db: Path) -> None:
  speaker_db.mkdir(parents=True, exist_ok=True)
  ref_text_path = speaker_db / "ref_text.txt"
  ref_text_path.write_text(ref_text or "")

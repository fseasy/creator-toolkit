from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

from .serialize import load_prompt_items

COMMON_GEN_KWARGS = dict(
  max_new_tokens=2048,
  do_sample=True,
  top_k=50,
  top_p=1.0,
  temperature=0.9,
  repetition_penalty=1.05,
  subtalker_dosample=True,
  subtalker_top_k=50,
  subtalker_top_p=1.0,
  subtalker_temperature=0.9,
)


@dataclass
class TTSEngine:
  model_path: Path
  device: str = "cuda:0"
  _model: Qwen3TTSModel | None = field(default=None, init=False)

  def load(self) -> None:
    if self._model is not None:
      return
    self._model = Qwen3TTSModel.from_pretrained(
      str(self.model_path),
      device_map=self.device,
      dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
      local_files_only=True,
    )

  @property
  def model(self) -> Qwen3TTSModel:
    if self._model is None:
      self.load()
    return self._model  # type: ignore[return-value]

  def create_voice_clone_prompt(
    self,
    ref_audio: str,
    ref_text: str | None,
    x_vector_only_mode: bool = False,
  ) -> list:
    return self.model.create_voice_clone_prompt(
      ref_audio=ref_audio,
      ref_text=ref_text,
      x_vector_only_mode=x_vector_only_mode,
    )

  def generate_voice_clone(
    self,
    text: list[str],
    language: list[str],
    voice_clone_prompt: list[VoiceClonePromptItem],
  ) -> tuple[list, int]:
    return self.model.generate_voice_clone(
      text=text,
      language=language,
      voice_clone_prompt=voice_clone_prompt,
      **COMMON_GEN_KWARGS,  # type: ignore
    )

  def load_prompt_from_db(self, speaker_db: Path) -> list:
    return load_prompt_items(speaker_db)

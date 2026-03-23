from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from qwen_tts import VoiceClonePromptItem


class SpeakerPromptCache:
  def __init__(self, max_size: int = 5) -> None:
    self._cache: OrderedDict[str, list[VoiceClonePromptItem]] = OrderedDict()
    self._max_size = max_size

  def get(self, speaker_name: str) -> list[VoiceClonePromptItem] | None:
    if speaker_name not in self._cache:
      return None
    self._cache.move_to_end(speaker_name)
    return self._cache[speaker_name]

  def put(self, speaker_name: str, prompt_items: list[VoiceClonePromptItem]) -> None:
    if speaker_name in self._cache:
      self._cache.move_to_end(speaker_name)
    else:
      if len(self._cache) >= self._max_size:
        self._cache.popitem(last=False)
    self._cache[speaker_name] = prompt_items

  def remove(self, speaker_name: str) -> None:
    self._cache.pop(speaker_name, None)

  def clear(self) -> None:
    self._cache.clear()

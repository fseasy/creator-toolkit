from enum import StrEnum

from pydantic import BaseModel, Field


class AudioFormat(StrEnum):
  WAV = "wav"
  MP3 = "mp3"


class CreateSpeakerResponse(BaseModel):
  speaker_name: str
  message: str


class BatchTTSRequest(BaseModel):
  speaker_name: str = Field(..., min_length=1)
  texts: list[str] = Field(..., min_length=1)
  languages: str | list[str] = Field(...)
  audio_fmt: AudioFormat = Field(default=AudioFormat.WAV)


class TTSRequest(BaseModel):
  speaker_name: str = Field(..., min_length=1)
  text: str
  language: str
  audio_fmt: AudioFormat = Field(default=AudioFormat.WAV)


class ManifestItem(BaseModel):
  text: str
  language: str
  audio_name: str


class Manifest(BaseModel):
  items: list[ManifestItem]

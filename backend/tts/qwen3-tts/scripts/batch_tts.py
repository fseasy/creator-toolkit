#!/usr/bin/env -S uv run python
from __future__ import annotations

import asyncio
from pathlib import Path

from fs_qwen3_tts_server import AudioFormat, Qwen3TTSClient


async def main() -> None:
  client = Qwen3TTSClient(base_url="http://localhost:17651")

  speaker_name = "dajuan_english"
  texts = [
    "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
  ]
  languages = ["English", "Chinese"]

  print(f"Running batch TTS for speaker '{speaker_name}'...")
  result = await client.batch_tts(
    speaker_name=speaker_name,
    texts=texts,
    languages=languages,
    audio_fmt=AudioFormat.MP3,
  )

  output_dir = Path("output")
  output_dir.mkdir(exist_ok=True)

  for i, item in enumerate(result.items):
    ext = "mp3" if item.audio_bytes[:2] == b"ID" else "wav"
    output_path = output_dir / f"audio_{i:04d}.{ext}"
    output_path.write_bytes(item.audio_bytes)
    print(f"  Saved: {output_path} ({len(item.audio_bytes)} bytes)")

  print(f"Done! {len(result.items)} audio files saved to {output_dir}/")


if __name__ == "__main__":
  asyncio.run(main())

#!/usr/bin/env -S uv run python
from __future__ import annotations

import asyncio
from pathlib import Path

from fs_qwen3_tts_server import AudioFormat, Qwen3TTSClient


async def main() -> None:
  client = Qwen3TTSClient(base_url="http://localhost:17651")

  speaker_name = "dajuan_english"
  text = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."

  language = "English"

  print(f"Running batch TTS for speaker '{speaker_name}'...")
  result = await client.tts(
    speaker_name=speaker_name,
    text=text,
    language=language,
    audio_fmt=AudioFormat.MP3,
  )

  output_dir = Path("output")
  output_dir.mkdir(exist_ok=True)

  output_path = output_dir / "tts_sample_output.mp3"
  output_path.write_bytes(result)
  print(f"  Saved: {output_path} ({len(result)} bytes)")

  print(f"Done! audio file saved to {output_path}")


if __name__ == "__main__":
  asyncio.run(main())

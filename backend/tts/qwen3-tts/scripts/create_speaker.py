#!/usr/bin/env -S uv run python
from __future__ import annotations

import asyncio
from pathlib import Path

from fs_qwen3_tts_server import Qwen3TTSClient


async def main() -> None:
  client = Qwen3TTSClient(base_url="http://localhost:17651")

  # speaker_name = "test_speaker"
  # ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
  # ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
  speaker_name = "dajuan_english"
  ref_audio = Path("./data/dajuan-english-sample.wav").resolve().absolute()
  ref_text = Path("./data/dajuan-english-sample.ref.txt").read_text(encoding="utf-8").strip()

  print(f"Creating speaker '{speaker_name}'...")
  response = await client.create_speaker(
    speaker_name=speaker_name,
    ref_audio=ref_audio,
    ref_text=ref_text,
  )
  print(f"Success: {response.message}")


if __name__ == "__main__":
  asyncio.run(main())

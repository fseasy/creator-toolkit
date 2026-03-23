#!/usr/bin/env -S uv run python
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from fs_qwen3_tts_server.routes import init_app, router

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "db"
MODEL_PATH = ROOT / "model-data" / "Qwen3-TTS-12Hz-0.6B-Base"


def create_app() -> FastAPI:
  app = FastAPI(
    title="Qwen3-TTS Server",
    description="Voice cloning and batch TTS API using Qwen3-TTS",
    version="0.1.0",
  )
  init_app(db_path=DB_PATH, model_path=MODEL_PATH)
  app.include_router(router)
  return app


if __name__ == "__main__":
  uvicorn.run(
    "main:create_app",
    factory=True,
    host="0.0.0.0",
    port=17651,
    reload=False,
  )

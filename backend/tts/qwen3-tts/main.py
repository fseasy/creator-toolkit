#!/usr/bin/env -S uv run python
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

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
    version="1.1.0",
  )
  init_app(db_path=DB_PATH, model_path=MODEL_PATH)
  app.include_router(router)

  app.add_exception_handler(HTTPException, _custom_http_exception_handler)  # type: ignore
  app.add_exception_handler(Exception, _universal_exception_handler)

  return app


async def _custom_http_exception_handler(request: Request, exc: HTTPException) -> Response:
  exc_info = True if exc.status_code >= 500 else None
  logger.log(logging.WARNING, f"HTTP Error: {exc.status_code} - {exc.detail}", exc_info=exc_info)

  # 返回给客户端原有的格式
  return JSONResponse(
    status_code=exc.status_code,
    content={"detail": exc.detail},
  )


async def _universal_exception_handler(request: Request, exc: Exception):
  # 这里建议用 logger.exception，因为这是未预料到的系统崩溃
  logger.exception(f"Unhandled System Error: {str(exc)}")
  return JSONResponse(
    status_code=500,
    content={"detail": f"Internal Server Error, exc={exc}"},
  )


if __name__ == "__main__":
  uvicorn.run(
    "main:create_app",
    factory=True,
    host="0.0.0.0",
    port=17651,
    reload=False,
  )

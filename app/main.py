from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin, documents, health, ingestion, knowledge_bases, rag, traces, voice
from app.container import Container
from app.core.config import Settings, get_settings
from app.core.errors import RAGError
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

VOICE_WEB_DIR = Path(__file__).resolve().parent / "voice" / "web"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        c = container or await Container.create(settings)
        app.state.container = c
        stop = asyncio.Event()
        worker_task = None
        if owned:
            await c.warmup()
            if settings.worker.embedded:
                worker_task = asyncio.create_task(c.worker.run_forever(stop), name="ingestion-worker")
        try:
            yield
        finally:
            stop.set()
            if worker_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait_for(worker_task, timeout=30)
            if owned:
                await c.aclose()

    app = FastAPI(
        title="Hybrid RAG Platform",
        version="0.1.0",
        description="Multi-tenant hybrid retrieval (Qdrant dense + PostgreSQL BM25) with RRF, "
        "cross-encoder reranking and grounded, cited generation.",
        lifespan=lifespan,
    )

    @app.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("request failed", extra={"code": exc.code, "error": exc.message})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    for module in (health, admin, knowledge_bases, documents, ingestion, rag, traces, voice):
        app.include_router(module.router)
    # Browser demo of the voice pipeline (mic -> VAD -> STT -> RAG -> TTS).
    app.mount("/voice", StaticFiles(directory=str(VOICE_WEB_DIR), html=True), name="voice-demo")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/voice/")

    return app


def app_factory() -> FastAPI:
    """ASGI entrypoint: ``uvicorn app.main:app_factory --factory``."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return create_app(settings)

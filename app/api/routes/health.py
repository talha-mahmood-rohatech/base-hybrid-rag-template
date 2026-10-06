from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import get_container
from app.container import Container

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(container: Container = Depends(get_container)) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        async with container.session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}"
    checks["qdrant"] = "ok" if await container.vector_store.healthcheck() else "error"
    healthy = all(v == "ok" for v in checks.values())
    s = container.settings
    body = {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
        "config": {
            "embedding": {
                "provider": s.embedding.provider,
                "model": s.embedding.model,
                "dimension": s.embedding.dimension,
                "version": s.embedding.version,
                "distance_metric": s.embedding.distance_metric,
            },
            "sparse": s.retrieval.sparse_provider,
            "fusion": {"strategy": s.retrieval.fusion, "rrf_k": s.retrieval.rrf_k},
            "reranker": {"provider": s.reranker.provider, "model": s.reranker.model},
            "llm": {"provider": s.llm.provider, "model": s.llm.model},
        },
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}

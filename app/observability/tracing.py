"""Persistence of per-request RAG traces (``retrieval_traces``)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import RetrievalTrace
from app.retrieval.types import Candidate, RetrievalResult

logger = logging.getLogger(__name__)


def result_rows(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": str(r.chunk_id), "document_id": str(r.document_id), "rank": r.rank, "score": r.score}
        for r in results
    ]


def candidate_rows(cands: list[Candidate]) -> list[dict[str, Any]]:
    return [c.explain() for c in cands]


@dataclass
class TraceData:
    id: uuid.UUID
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID | None
    query: str
    filters: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    dense_results: list[dict] = field(default_factory=list)
    sparse_results: list[dict] = field(default_factory=list)
    fused_results: list[dict] = field(default_factory=list)
    reranked_results: list[dict] = field(default_factory=list)
    selected_chunks: list[dict] = field(default_factory=list)
    dropped_chunks: list[dict] = field(default_factory=list)
    context_tokens: int = 0
    llm: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    timings_ms: dict[str, Any] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    status: str = "ok"
    total_latency_ms: float = 0.0


class TraceRecorder:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def save(self, data: TraceData) -> None:
        try:
            async with self.session_factory() as session, session.begin():
                session.add(
                    RetrievalTrace(
                        id=data.id,
                        tenant_id=data.tenant_id,
                        knowledge_base_id=data.knowledge_base_id,
                        status=data.status,
                        query=data.query,
                        filters=data.filters,
                        config=data.config,
                        dense_results=data.dense_results,
                        sparse_results=data.sparse_results,
                        fused_results=data.fused_results,
                        reranked_results=data.reranked_results,
                        selected_chunks=data.selected_chunks,
                        dropped_chunks=data.dropped_chunks,
                        context_tokens=data.context_tokens,
                        llm=data.llm,
                        answer=data.answer,
                        citations=data.citations,
                        timings_ms=data.timings_ms,
                        errors=data.errors,
                        total_latency_ms=data.total_latency_ms,
                    )
                )
        except Exception:  # tracing must never break the request path
            logger.exception("failed to persist trace", extra={"trace_id": str(data.id)})

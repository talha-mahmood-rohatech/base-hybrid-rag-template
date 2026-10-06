"""Common retrieval data types shared by dense/sparse retrievers, fusion and reranking."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.filters import FilterCondition


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    """Hard isolation boundary. Always applied; never derived from user-provided filters."""

    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    active_only: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    scope: RetrievalScope
    filters: tuple[FilterCondition, ...] = ()
    top_k: int = 50


@dataclass(slots=True)
class RetrievalResult:
    """One hit from one retriever. ``rank`` is 1-based within that retriever's list."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float
    rank: int
    retriever: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkView:
    """Hydrated chunk (from PostgreSQL, the source of truth for text/metadata)."""

    chunk_id: uuid.UUID
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_name: str
    document_type: str
    source: str | None
    chunk_index: int
    text: str
    token_count: int
    page: int | None
    page_end: int | None
    section: str | None
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    """A chunk moving through fusion -> reranking -> context building."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    fused_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    fused_rank: int = 0
    reranker_score: float | None = None
    final_rank: int | None = None
    chunk: ChunkView | None = None

    @property
    def dense_rank(self) -> int | None:
        return self.ranks.get("dense")

    @property
    def sparse_rank(self) -> int | None:
        return self.ranks.get("sparse")

    @property
    def rrf_score(self) -> float:
        return self.fused_score

    def explain(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "dense_rank": self.dense_rank,
            "dense_score": self.scores.get("dense"),
            "sparse_rank": self.sparse_rank,
            "sparse_score": self.scores.get("sparse"),
            "fused_rank": self.fused_rank,
            "rrf_score": self.fused_score,
            "reranker_score": self.reranker_score,
            "final_rank": self.final_rank,
        }
        if self.chunk is not None:
            out.update(
                document_name=self.chunk.document_name,
                page=self.chunk.page,
                section=self.chunk.section,
                preview=self.chunk.text[:200],
            )
        return out

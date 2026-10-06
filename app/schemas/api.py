"""Pydantic v2 request/response models for the public API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- tenants (admin) ----------------------------------------------------------------
class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9\-]*$")


class TenantCreated(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    api_key: str = Field(description="Shown once. Store it securely.")


# --- knowledge bases ----------------------------------------------------------------
class ChunkingOverrides(BaseModel):
    chunk_size: int | None = Field(default=None, gt=0, le=8192)
    chunk_overlap: int | None = Field(default=None, ge=0)
    unit: Literal["tokens", "characters"] | None = None


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    fts_language: str = Field(default="english", pattern=r"^[a-z_]+$")
    chunking: ChunkingOverrides | None = None
    embedding_config_id: uuid.UUID | None = Field(
        default=None, description="Pin a specific embedding configuration; defaults to the platform default."
    )


class EmbeddingConfigOut(ORMModel):
    id: uuid.UUID
    provider: str
    model: str
    dimension: int
    version: str
    distance_metric: str
    collection_name: str


class KnowledgeBaseOut(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    fts_language: str
    settings: dict[str, Any]
    embedding_config: EmbeddingConfigOut
    created_at: datetime


# --- documents ----------------------------------------------------------------------
class DocumentVersionOut(BaseModel):
    id: uuid.UUID
    version_number: int
    checksum: str
    size_bytes: int
    status: str
    is_active: bool
    chunk_count: int
    page_count: int | None
    title: str | None
    error: str | None
    created_at: datetime
    ingested_at: datetime | None


class IngestionJobOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    status: str
    attempts: int
    max_attempts: int
    error: str | None
    stats: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class DocumentOut(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    external_id: str
    name: str
    document_type: str
    source: str | None
    metadata: dict[str, Any]
    permissions: list[str]
    active_version_id: uuid.UUID | None
    versions: list[DocumentVersionOut] = []


class DocumentUploadOut(BaseModel):
    outcome: Literal["created", "new_version", "unchanged", "requeued"]
    document: DocumentOut
    version: DocumentVersionOut
    job: IngestionJobOut | None


class IngestionJobCreate(BaseModel):
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None


# --- query --------------------------------------------------------------------------
class RetrievalOptions(BaseModel):
    dense_top_k: int | None = Field(default=None, ge=1, le=500)
    sparse_top_k: int | None = Field(default=None, ge=1, le=500)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    rrf_top_k: int | None = Field(default=None, ge=1, le=500)
    fusion: Literal["rrf", "relative_score"] | None = None
    rerank: bool = True
    dense: bool = Field(default=True, description="Use the dense (Qdrant) retriever")
    sparse: bool = Field(default=True, description="Use the sparse (BM25) retriever")

    @model_validator(mode="after")
    def _one_retriever(self) -> RetrievalOptions:
        if not (self.dense or self.sparse):
            raise ValueError("at least one of 'dense' or 'sparse' must be enabled")
        return self


class QueryRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    query: str = Field(min_length=1, max_length=4000)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=8, ge=1, le=50)
    generate_answer: bool = True
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)
    include_results: bool = True


class CitationOut(BaseModel):
    citation_id: int
    document_id: str
    document_name: str
    document_version_id: str
    chunk_id: str
    page: int | None
    page_end: int | None
    section: str | None
    source: str | None
    snippet: str
    reranker_score: float | None
    rrf_score: float


class RetrievedChunkOut(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    section: str | None
    text: str
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    reranker_score: float | None


class RetrievalCounts(BaseModel):
    dense_candidates: int
    sparse_candidates: int
    rrf_candidates: int
    reranked_candidates: int
    context_chunks: int


class UsageOut(BaseModel):
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[CitationOut]
    trace_id: uuid.UUID
    retrieval: RetrievalCounts
    results: list[RetrievedChunkOut] = []
    usage: UsageOut | None = None
    degraded: bool = False
    latency_ms: float


class TraceOut(ORMModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID | None
    status: str
    query: str
    filters: dict[str, Any]
    config: dict[str, Any]
    dense_results: list[Any]
    sparse_results: list[Any]
    fused_results: list[Any]
    reranked_results: list[Any]
    selected_chunks: list[Any]
    dropped_chunks: list[Any]
    context_tokens: int
    llm: dict[str, Any]
    answer: str | None
    citations: list[Any]
    timings_ms: dict[str, Any]
    errors: list[Any]
    total_latency_ms: float
    created_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    error: ErrorBody

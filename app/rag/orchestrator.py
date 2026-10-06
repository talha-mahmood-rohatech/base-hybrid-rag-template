"""RAG orchestrator: Query -> (Dense || Sparse) -> Fusion -> Hydrate -> Rerank -> Context -> LLM -> Citations.

Composes provider *interfaces* only; concrete providers are injected by the container.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import RetrievalSettings
from app.generation.citations import Citation, extract_citations
from app.generation.context import BuiltContext, ContextBuilder
from app.generation.prompts import NO_ANSWER, build_messages
from app.ingestion.embedding_configs import spec_of
from app.ingestion.service import get_kb
from app.models import EmbeddingConfig
from app.observability.tracing import TraceData, TraceRecorder, candidate_rows, result_rows
from app.providers.llms.base import LLMProvider, LLMResponse
from app.providers.registry import EmbeddingProviderCache, build_fusion
from app.providers.sparse.base import SparseSearch
from app.providers.vectorstores.base import VectorStore
from app.retrieval.dense import DenseRetriever
from app.retrieval.filters import parse_filters
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.repository import ChunkRepository
from app.retrieval.reranking.stage import RerankingStage
from app.retrieval.sparse import SparseRetriever
from app.retrieval.types import Candidate, RetrievalQuery, RetrievalScope

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryOptions:
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    rrf_top_k: int
    top_k: int
    fusion: str = "rrf"
    rerank: bool = True
    generate_answer: bool = True
    use_dense: bool = True
    use_sparse: bool = True

    @classmethod
    def from_settings(cls, s: RetrievalSettings, **overrides: Any) -> QueryOptions:
        base = cls(
            dense_top_k=s.dense_top_k,
            sparse_top_k=s.sparse_top_k,
            rrf_k=s.rrf_k,
            rrf_top_k=s.rrf_top_k,
            top_k=s.final_top_k,
            fusion=s.fusion,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(base, k, v)
        return base


@dataclass(slots=True)
class RAGResult:
    trace_id: uuid.UUID
    answer: str | None
    citations: list[Citation]
    results: list[Candidate]
    counts: dict[str, int]
    fused: list[Candidate]
    context: BuiltContext | None
    llm: LLMResponse | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.errors)


class RAGOrchestrator:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        vector_store: VectorStore,
        embedders: EmbeddingProviderCache,
        sparse_search: SparseSearch,
        reranking: RerankingStage,
        context_builder: ContextBuilder,
        llm: LLMProvider,
        traces: TraceRecorder,
        retrieval_settings: RetrievalSettings,
        llm_max_tokens: int = 800,
        llm_temperature: float = 0.0,
    ) -> None:
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.embedders = embedders
        self.sparse_search = sparse_search
        self.reranking = reranking
        self.context_builder = context_builder
        self.llm = llm
        self.traces = traces
        self.retrieval_settings = retrieval_settings
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.chunks = ChunkRepository()

    async def query(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        query: str,
        filters: dict[str, Any] | None,
        options: QueryOptions,
    ) -> RAGResult:
        t_start = time.perf_counter()
        trace = TraceData(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            query=query,
            filters=filters or {},
        )
        try:
            result = await self._run(trace, tenant_id, knowledge_base_id, query, filters, options)
        except Exception as exc:
            trace.status = "error"
            trace.errors.append({"stage": "pipeline", "error": f"{type(exc).__name__}: {exc}"})
            trace.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            if getattr(exc, "status_code", 500) == 404:
                trace.knowledge_base_id = None  # don't reference other tenants' KB ids
            await self.traces.save(trace)
            raise
        trace.total_latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        trace.timings_ms["total_ms"] = trace.total_latency_ms
        result.timings_ms = trace.timings_ms
        await self.traces.save(trace)
        return result

    async def _run(
        self,
        trace: TraceData,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        query: str,
        raw_filters: dict[str, Any] | None,
        options: QueryOptions,
    ) -> RAGResult:
        timings = trace.timings_ms
        async with self.session_factory() as session:
            kb = await get_kb(session, tenant_id, kb_id)
            emb_cfg = await session.get(EmbeddingConfig, kb.embedding_config_id)
            assert emb_cfg is not None
            fts_language = kb.fts_language
        filters = parse_filters(raw_filters)
        scope = RetrievalScope(tenant_id=tenant_id, knowledge_base_id=kb_id)

        embedder = self.embedders.get(spec_of(emb_cfg), emb_cfg.params)
        fusion = build_fusion(options.fusion, options.rrf_k)
        trace.config = {
            "dense_top_k": options.dense_top_k,
            "sparse_top_k": options.sparse_top_k,
            "fusion": {"strategy": fusion.name, **fusion.params()},
            "rrf_k": options.rrf_k,
            "rrf_top_k": options.rrf_top_k,
            "top_k": options.top_k,
            "embedding": {
                "provider": emb_cfg.provider,
                "model": emb_cfg.model,
                "dimension": emb_cfg.dimension,
                "version": emb_cfg.version,
                "distance_metric": emb_cfg.distance_metric,
                "collection": emb_cfg.collection_name,
            },
            "sparse": {"provider": type(self.sparse_search).__name__, "language": fts_language},
            "reranker": {
                "provider": self.reranking.reranker.provider,
                "model": self.reranking.reranker.model,
                "enabled": options.rerank and self.reranking.enabled,
            },
            "generate_answer": options.generate_answer,
            "retrievers": [
                n for n, on in (("dense", options.use_dense), ("sparse", options.use_sparse)) if on
            ],
        }

        hybrid = HybridRetriever(
            DenseRetriever(embedder, self.vector_store, emb_cfg.collection_name),
            SparseRetriever(self.sparse_search, fts_language),
            fusion,
            tolerate_failure=self.retrieval_settings.tolerate_retriever_failure,
        )
        hr = await hybrid.retrieve(
            RetrievalQuery(query, scope, filters, options.dense_top_k),
            RetrievalQuery(query, scope, filters, options.sparse_top_k),
            fused_top_k=options.rrf_top_k,
            use_dense=options.use_dense,
            use_sparse=options.use_sparse,
        )
        timings.update(hr.timings_ms)
        trace.errors.extend(hr.errors)
        trace.dense_results = result_rows(hr.dense)
        trace.sparse_results = result_rows(hr.sparse)

        # Hydrate from PostgreSQL; candidates that are no longer active are dropped here.
        t = time.perf_counter()
        async with self.session_factory() as session:
            views = await self.chunks.get_many(session, scope, [c.chunk_id for c in hr.fused])
        fused: list[Candidate] = []
        for c in hr.fused:
            c.chunk = views.get(c.chunk_id)
            if c.chunk is None:
                trace.dropped_chunks.append({"chunk_id": str(c.chunk_id), "reason": "inactive_or_missing"})
            else:
                fused.append(c)
        timings["hydrate_ms"] = round((time.perf_counter() - t) * 1000, 2)
        trace.fused_results = candidate_rows(fused)

        t = time.perf_counter()
        reranked = await self.reranking.run(query, fused, options.top_k, enabled=options.rerank)
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000, 2)
        trace.reranked_results = candidate_rows(reranked)

        t = time.perf_counter()
        context = self.context_builder.build(reranked)
        timings["context_ms"] = round((time.perf_counter() - t) * 1000, 2)
        trace.context_tokens = context.token_count
        trace.dropped_chunks.extend(context.dropped)
        trace.selected_chunks = [
            {
                "citation_id": b.citation_id,
                "tokens": b.tokens,
                "why": {
                    "dense_rank": b.candidate.dense_rank,
                    "sparse_rank": b.candidate.sparse_rank,
                    "rrf_score": b.candidate.fused_score,
                    "fused_rank": b.candidate.fused_rank,
                    "reranker_score": b.candidate.reranker_score,
                    "final_rank": b.candidate.final_rank,
                },
                **b.candidate.explain(),
            }
            for b in context.blocks
        ]

        answer: str | None = None
        citations: list[Citation] = []
        llm_resp: LLMResponse | None = None
        if options.generate_answer:
            if not context.blocks:
                answer = NO_ANSWER
                trace.llm = {"skipped": "no_context"}
            else:
                t = time.perf_counter()
                llm_resp = await self.llm.generate(
                    build_messages(query, context.text),
                    max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature,
                )
                timings["generate_ms"] = round((time.perf_counter() - t) * 1000, 2)
                cited = extract_citations(llm_resp.text, context)
                answer, citations = cited.answer, cited.citations
                trace.llm = {
                    "provider": llm_resp.provider,
                    "model": llm_resp.model,
                    "usage": {
                        "prompt_tokens": llm_resp.usage.prompt_tokens,
                        "completion_tokens": llm_resp.usage.completion_tokens,
                        "total_tokens": llm_resp.usage.total_tokens,
                    },
                    "latency_ms": llm_resp.latency_ms,
                    "finish_reason": llm_resp.finish_reason,
                    "raw_answer": llm_resp.text,
                    "invalid_citation_ids": cited.invalid_ids,
                }
            trace.answer = answer
            trace.citations = [c.to_dict() for c in citations]

        counts = {
            "dense_candidates": len(hr.dense),
            "sparse_candidates": len(hr.sparse),
            "rrf_candidates": len(fused),
            "reranked_candidates": len(reranked),
            "context_chunks": len(context.blocks),
        }
        trace.timings_ms["counts"] = counts
        return RAGResult(
            trace_id=trace.id,
            answer=answer,
            citations=citations,
            results=reranked,
            counts=counts,
            fused=fused,
            context=context,
            llm=llm_resp,
            errors=list(trace.errors),
        )

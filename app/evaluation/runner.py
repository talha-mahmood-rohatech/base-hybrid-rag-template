"""Runs an evaluation dataset against one or more pipeline variants and compares them.

Each variant gets an isolated, temporary tenant + knowledge base (pinned to the variant's
embedding configuration), so embedding models, rerankers, chunking and fusion parameters
can be compared on identical data. Temporary data is removed afterwards.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from app.container import Container
from app.core.config import Settings
from app.evaluation.dataset import EvalDataset, EvalQuery, Variant, VariantSet
from app.evaluation.judges import HeuristicJudge, Judge
from app.evaluation.metrics import (
    citation_precision,
    citation_validity,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.ingestion.embedding_configs import collection_of, spec_of
from app.models import KnowledgeBase, Tenant
from app.rag.orchestrator import QueryOptions, RAGOrchestrator
from app.retrieval.types import Candidate

logger = logging.getLogger(__name__)


@dataclass
class VariantReport:
    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    per_query: list[dict[str, Any]] = field(default_factory=list)
    ingest_seconds: float = 0.0
    query_latency_ms_p50: float = 0.0


def passage_ids(q: EvalQuery, candidates: list[Candidate], doc_ext_ids: dict[uuid.UUID, str]) -> list[str]:
    """Map each retrieved chunk to the first relevant passage it satisfies (or its own id)."""
    out = []
    for c in candidates:
        assert c.chunk is not None
        ext = doc_ext_ids.get(c.document_id)
        pid = str(c.chunk_id)
        for i, passage in enumerate(q.relevant):
            if passage.external_id == ext and (
                passage.contains is None or passage.contains.lower() in c.chunk.text.lower()
            ):
                pid = f"passage:{i}"
                break
        out.append(pid)
    # A passage counts once (first occurrence); later duplicates become non-relevant fillers.
    seen: set[str] = set()
    dedup = []
    for i, pid in enumerate(out):
        dedup.append(pid if pid not in seen else f"dup:{i}")
        seen.add(pid)
    return dedup


class EvaluationRunner:
    def __init__(self, settings: Settings, *, judge_factory=None) -> None:
        self.settings = settings
        self.judge_factory = judge_factory

    def _variant_settings(self, v: Variant) -> Settings:
        data = self.settings.model_dump()
        for section in ("embedding", "reranker", "chunking", "retrieval"):
            data[section] = {**data[section], **getattr(v, section)}
        return Settings.model_validate(data)

    async def run(self, dataset: EvalDataset, variants: VariantSet) -> list[VariantReport]:
        reports = []
        for v in variants.variants:
            logger.info("evaluating variant", extra={"variant": v.name})
            reports.append(await self._run_variant(dataset, v, variants.k_values))
        return reports

    async def _run_variant(self, dataset: EvalDataset, v: Variant, ks: list[int]) -> VariantReport:
        settings = self._variant_settings(v)
        container = await Container.create(settings, initialize=False)
        tenant_id = uuid.uuid4()
        try:
            async with container.session_factory() as s, s.begin():
                emb = await container.embedding_configs.get_or_create(
                    s,
                    provider=settings.embedding.provider,
                    model=settings.embedding.model,
                    dimension=settings.embedding.dimension,
                    version=settings.embedding.version,
                    distance_metric=settings.embedding.distance_metric,
                )
                s.add(
                    Tenant(
                        id=tenant_id, name=f"eval {v.name}", slug=f"eval-{tenant_id.hex[:12]}", is_active=True
                    )
                )
                await s.flush()
                kb = KnowledgeBase(
                    id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    name=f"eval-{v.name}",
                    embedding_config_id=emb.id,
                    fts_language=settings.retrieval.fts_language,
                    settings={"chunking": v.chunking} if v.chunking else {},
                )
                s.add(kb)
            await container.vector_store.ensure_collection(collection_of(emb))

            # --- ingest (direct pipeline call, bypassing the shared job queue)
            t0 = time.perf_counter()
            doc_ext_ids: dict[uuid.UUID, str] = {}
            for d in dataset.documents:
                filename, data = d.load(dataset.base_dir)
                async with container.session_factory() as s, s.begin():
                    res = await container.documents.upload(
                        s,
                        tenant_id=tenant_id,
                        knowledge_base_id=kb.id,
                        data=data,
                        filename=filename,
                        external_id=d.external_id,
                        metadata=d.metadata,
                        enqueue=False,
                    )
                doc_ext_ids[res.document.id] = d.external_id
                await container.pipeline.run(res.version.id)
            ingest_s = time.perf_counter() - t0

            embedder = container.embedders.get(spec_of(emb), emb.params)
            judge: Judge = self.judge_factory(container) if self.judge_factory else HeuristicJudge(embedder)
            orch: RAGOrchestrator = container.orchestrator  # built from the variant's settings

            per_query, latencies = [], []
            for q in dataset.queries:
                opts = QueryOptions.from_settings(
                    settings.retrieval,
                    top_k=v.top_k,
                    rerank=v.rerank,
                    generate_answer=v.generate_answer,
                    use_dense=v.use_dense,
                    use_sparse=v.use_sparse,
                )
                t = time.perf_counter()
                result = await orch.query(
                    tenant_id=tenant_id,
                    knowledge_base_id=kb.id,
                    query=q.query,
                    filters=q.filters,
                    options=opts,
                )
                latencies.append((time.perf_counter() - t) * 1000)
                per_query.append(await self._score_query(q, result, doc_ext_ids, ks, judge, v))

            metrics = {
                key: round(statistics.fmean(pq[key] for pq in per_query), 4)
                for key in per_query[0]
                if isinstance(per_query[0][key], int | float) and not isinstance(per_query[0][key], bool)
            }
            return VariantReport(
                name=v.name,
                config={
                    "embedding": f"{emb.provider}:{emb.model} ({emb.dimension}d)",
                    "reranker": f"{settings.reranker.provider}:{settings.reranker.model}"
                    if v.rerank
                    else "off",
                    "retrievers": "+".join(
                        n for n, on in (("dense", v.use_dense), ("sparse", v.use_sparse)) if on
                    ),
                    "fusion": settings.retrieval.fusion,
                    "rrf_k": settings.retrieval.rrf_k,
                    "dense_top_k": settings.retrieval.dense_top_k,
                    "sparse_top_k": settings.retrieval.sparse_top_k,
                    "rrf_top_k": settings.retrieval.rrf_top_k,
                    "top_k": v.top_k,
                    "chunking": {**settings.chunking.model_dump(), **v.chunking},
                },
                metrics=metrics,
                per_query=per_query,
                ingest_seconds=round(ingest_s, 2),
                query_latency_ms_p50=round(statistics.median(latencies), 1),
            )
        finally:
            await self._cleanup(container, tenant_id)
            await container.aclose()

    async def _score_query(self, q, result, doc_ext_ids, ks, judge: Judge, v: Variant) -> dict[str, Any]:
        n_passages = len(q.relevant)
        relevant = {f"passage:{i}": p.grade for i, p in enumerate(q.relevant)}
        final_ids = passage_ids(q, result.results, doc_ext_ids)
        fused_ids = passage_ids(q, result.fused, doc_ext_ids)
        row: dict[str, Any] = {"id": q.id, "query": q.query, "n_relevant_passages": n_passages}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(final_ids, relevant, k)
            row[f"precision@{k}"] = precision_at_k(final_ids, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(final_ids, relevant, k)
            row[f"hit@{k}"] = hit_rate_at_k(final_ids, relevant, k)
        row["mrr"] = mrr(final_ids, relevant)
        row["fused_mrr"] = mrr(fused_ids, relevant)  # before reranking -> shows reranker uplift
        row["fused_recall@40"] = recall_at_k(fused_ids, relevant, 40)
        if v.generate_answer and result.answer is not None:
            context_ids = [
                str(b.candidate.chunk_id) for b in (result.context.blocks if result.context else [])
            ]
            cited = [c.chunk_id for c in result.citations]
            cited_passages = (
                passage_ids(
                    q,
                    [b.candidate for b in result.context.blocks if str(b.candidate.chunk_id) in cited],
                    doc_ext_ids,
                )
                if result.context
                else []
            )
            row["citation_precision"] = citation_precision(cited_passages, relevant) if cited else 0.0
            row["citation_validity"] = citation_validity(cited, context_ids)
            contexts = [b.text for b in result.context.blocks] if result.context else []
            row["faithfulness"] = await judge.faithfulness(result.answer, contexts)
            row["answer_relevance"] = await judge.answer_relevance(q.query, result.answer)
            row["answer"] = result.answer
        row["top_chunks"] = [
            {
                "rank": c.final_rank,
                "document_id": str(c.document_id),
                "rrf": round(c.fused_score, 5),
                "rerank": c.reranker_score,
                "text": c.chunk.text[:120] if c.chunk else "",
            }
            for c in result.results[:3]
        ]
        return row

    async def _cleanup(self, container: Container, tenant_id: uuid.UUID) -> None:
        try:
            client = container.vector_store
            async with container.session_factory() as s:
                from sqlalchemy import select

                from app.models import EmbeddingConfig

                cols = [r.collection_name for r in (await s.execute(select(EmbeddingConfig))).scalars()]
            for col in cols:
                await client.delete_by_tenant(col, tenant_id)  # type: ignore[attr-defined]
            async with container.session_factory() as s, s.begin():
                await s.execute(delete(Tenant).where(Tenant.id == tenant_id))  # cascades
        except Exception:
            logger.exception("eval cleanup failed", extra={"tenant_id": str(tenant_id)})


def write_reports(
    reports: list[VariantReport], dataset: EvalDataset, out_dir: str | Path, ks: list[int]
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / f"eval-{stamp}.json").write_text(
        json.dumps(
            {"dataset": dataset.name, "created_at": stamp, "variants": [r.__dict__ for r in reports]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    cols = [f"recall@{ks[-1]}", f"precision@{ks[0]}", "mrr", f"ndcg@{ks[-1]}", "fused_mrr"]
    extra = [
        c
        for c in ("faithfulness", "answer_relevance", "citation_precision", "citation_validity")
        if any(c in r.metrics for r in reports)
    ]
    header = ["variant", "embedding", "reranker", "retrievers", "rrf_k", *cols, *extra, "p50 ms"]
    lines = [
        f"# Evaluation: {dataset.name} ({stamp})",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for r in reports:
        vals = [
            r.name,
            r.config["embedding"],
            r.config["reranker"],
            r.config["retrievers"],
            str(r.config["rrf_k"]),
        ]
        vals += [f"{r.metrics[c]:.3f}" if c in r.metrics else "n/a" for c in cols + extra]
        vals.append(f"{r.query_latency_ms_p50:.0f}")
        lines.append("| " + " | ".join(vals) + " |")
    md = out / f"eval-{stamp}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md

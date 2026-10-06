import asyncio
import time
import uuid

import pytest

from app.ingestion.embedding_configs import spec_of
from app.providers.sparse.postgres import PostgresBM25Search
from app.retrieval.dense import DenseRetrieval, DenseRetriever
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.types import RetrievalQuery, RetrievalResult, RetrievalScope
from tests.integration.helpers import ingest, make_workspace

DOCS = {
    "refund_policy.md": "# Refund Policy\n\n## Eligibility\n\nMonthly plans are refundable within 30 days of purchase.\n\n"
    "## Exceptions\n\nUsage-based charges are never refundable.",
    "sla.md": "# SLA\n\nUptime commitment is 99.95% per month. Service credits apply below that.",
    "faq.txt": "API rate limits are 1,000 requests per minute on the Standard plan.",
}


@pytest.fixture(scope="module")
async def ws(container):
    w = await make_workspace(container)
    for name, body in DOCS.items():
        await ingest(container, w, name, body)
    return w


async def _hybrid(container, ws):
    async with container.session_factory() as s:
        emb = await container.embedding_configs.get_default(s)
    dense = DenseRetriever(container.embedders.get(spec_of(emb)), container.vector_store, emb.collection_name)
    sparse = SparseRetriever(PostgresBM25Search(container.session_factory))
    return HybridRetriever(dense, sparse, ReciprocalRankFusion(k=60))


@pytest.mark.integration
async def test_hybrid_fuses_both_paths_with_exact_rrf(container, ws):
    hybrid = await _hybrid(container, ws)
    scope = RetrievalScope(ws.tenant_id, ws.kb_id)
    q = "are usage-based charges refundable?"
    res = await hybrid.retrieve(RetrievalQuery(q, scope, top_k=50), RetrievalQuery(q, scope, top_k=50), 40)
    assert res.dense and res.sparse and not res.errors
    dense_rank = {r.chunk_id: r.rank for r in res.dense}
    sparse_rank = {r.chunk_id: r.rank for r in res.sparse}
    assert {c.chunk_id for c in res.fused} == set(dense_rank) | set(sparse_rank)
    for c in res.fused:
        expected = sum(1 / (60 + r) for r in (dense_rank.get(c.chunk_id), sparse_rank.get(c.chunk_id)) if r)
        assert c.rrf_score == pytest.approx(expected)
        assert c.dense_rank == dense_rank.get(c.chunk_id)
        assert c.sparse_rank == sparse_rank.get(c.chunk_id)
    assert res.timings_ms["fusion_ms"] >= 0


class _Slow:
    def __init__(self, delay, out):
        self.delay, self.out = delay, out

    async def retrieve(self, q):
        await asyncio.sleep(self.delay)
        return self.out


def _r(name):
    return RetrievalResult(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, name),
        document_id=uuid.uuid4(),
        score=1.0,
        rank=1,
        retriever="x",
    )


async def test_dense_and_sparse_run_concurrently():
    dense = _Slow(0.3, DenseRetrieval([_r("a")], 0.0, 0.0))
    sparse = _Slow(0.3, [_r("b")])
    hybrid = HybridRetriever(dense, sparse, ReciprocalRankFusion())  # type: ignore[arg-type]
    q = RetrievalQuery("q", RetrievalScope(uuid.uuid4(), uuid.uuid4()))
    t = time.perf_counter()
    res = await hybrid.retrieve(q, q, 10)
    elapsed = time.perf_counter() - t
    assert elapsed < 0.5, f"retrievers ran sequentially ({elapsed:.2f}s)"
    assert len(res.fused) == 2


async def test_degraded_mode_when_one_retriever_fails():
    class Boom:
        async def retrieve(self, q):
            raise RuntimeError("qdrant down")

    hybrid = HybridRetriever(Boom(), _Slow(0, [_r("b")]), ReciprocalRankFusion())  # type: ignore[arg-type]
    q = RetrievalQuery("q", RetrievalScope(uuid.uuid4(), uuid.uuid4()))
    res = await hybrid.retrieve(q, q, 10)
    assert res.degraded and res.errors[0]["stage"] == "dense"
    assert len(res.fused) == 1

    strict = HybridRetriever(Boom(), _Slow(0, []), ReciprocalRankFusion(), tolerate_failure=False)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        await strict.retrieve(q, q, 10)

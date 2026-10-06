"""PostgreSQL BM25 / FTS sparse retrieval, verified against an independent Python BM25."""

import math

import pytest
from sqlalchemy import text

from app.providers.sparse.postgres import PostgresBM25Search, PostgresFTSSearch, query_terms
from app.retrieval.filters import parse_filters
from app.retrieval.types import RetrievalQuery, RetrievalScope
from tests.integration.helpers import ingest, make_workspace

pytestmark = pytest.mark.integration

DOCS = {
    "refunds.md": "# Refunds\n\nRefunds are available within 30 days. Refunds for annual plans are prorated.",
    "shipping.md": "# Shipping\n\nShipping takes five business days. Express shipping is available.",
    "security.md": "# Security\n\nReport security incidents within 24 hours. Refunds are unrelated to security.",
    "billing.txt": "Billing questions: invoices are sent monthly. Refund requests go to billing.",
}


@pytest.fixture(scope="module")
async def ws(container):
    w = await make_workspace(container)
    for name, body in DOCS.items():
        await ingest(container, w, name, body, metadata={"team": "support" if "ship" in name else "finance"})
    return w


async def python_bm25(container, ws, query: str, k1=1.2, b=0.75) -> dict:
    """Reference implementation reading raw tsvectors from PostgreSQL."""
    async with container.session_factory() as s:
        terms = await query_terms(s, query, "english")
        rows = (
            await s.execute(
                text(
                    "SELECT c.id, c.token_count, (SELECT jsonb_object_agg(u.lexeme, coalesce(array_length(u.positions,1),1)) "
                    "FROM unnest(c.search_vector) u) AS tf FROM chunks c "
                    "WHERE c.tenant_id = :t AND c.knowledge_base_id = :k AND c.is_active"
                ),
                {"t": ws.tenant_id, "k": ws.kb_id},
            )
        ).all()
    n = len(rows)
    avgdl = sum(r.token_count for r in rows) / n
    df = {t: sum(1 for r in rows if t in (r.tf or {})) for t in terms}
    scores = {}
    for r in rows:
        tf = r.tf or {}
        s = 0.0
        for t in terms:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * r.token_count / avgdl))
        if s > 0:
            scores[r.id] = s
    return scores


async def test_bm25_matches_reference_implementation(container, ws):
    query = "how do refunds work for annual plans?"
    search = PostgresBM25Search(container.session_factory)
    results = await search.search(RetrievalQuery(query, RetrievalScope(ws.tenant_id, ws.kb_id), top_k=50))
    expected = await python_bm25(container, ws, query)

    assert {r.chunk_id for r in results} == set(expected)
    for r in results:
        assert r.score == pytest.approx(expected[r.chunk_id], rel=1e-9)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


async def test_bm25_ranks_most_relevant_first(container, ws):
    search = PostgresBM25Search(container.session_factory)
    res = await search.search(RetrievalQuery("express shipping", RetrievalScope(ws.tenant_id, ws.kb_id)))
    async with container.session_factory() as s:
        top_text = (
            await s.execute(text("SELECT text FROM chunks WHERE id=:i"), {"i": res[0].chunk_id})
        ).scalar()
    assert "Express shipping" in top_text


async def test_stopword_only_query_returns_nothing(container, ws):
    search = PostgresBM25Search(container.session_factory)
    assert await search.search(RetrievalQuery("the and of", RetrievalScope(ws.tenant_id, ws.kb_id))) == []


async def test_sparse_metadata_and_type_filters(container, ws):
    search = PostgresBM25Search(container.session_factory)
    scope = RetrievalScope(ws.tenant_id, ws.kb_id)
    all_hits = await search.search(RetrievalQuery("refunds", scope))
    finance = await search.search(
        RetrievalQuery("refunds", scope, parse_filters({"metadata.team": "finance"}))
    )
    txt_only = await search.search(RetrievalQuery("refund", scope, parse_filters({"document_type": "txt"})))
    assert len(all_hits) >= 3
    assert {r.chunk_id for r in finance} <= {r.chunk_id for r in all_hits}
    assert len(txt_only) == 1
    none = await search.search(RetrievalQuery("refunds", scope, parse_filters({"metadata.team": "nobody"})))
    assert none == []


async def test_fts_variant(container, ws):
    res = await PostgresFTSSearch(container.session_factory).search(
        RetrievalQuery("security incidents", RetrievalScope(ws.tenant_id, ws.kb_id))
    )
    assert res and res[0].score > 0

"""Cross-tenant retrieval must be impossible - at the store, orchestrator and API level."""

import httpx
import pytest

from app.ingestion.embedding_configs import spec_of
from app.main import create_app
from app.providers.sparse.postgres import PostgresBM25Search
from app.rag.orchestrator import QueryOptions
from app.retrieval.types import RetrievalQuery, RetrievalScope
from tests.integration.helpers import ingest, make_workspace

pytestmark = pytest.mark.integration

SECRET_A = "# Payroll\n\nAcme's secret payroll bonus code is ALPHA-7781 for all engineers."
SECRET_B = "# Payroll\n\nGlobex's secret payroll bonus code is BRAVO-4410 for all engineers."


@pytest.fixture(scope="module")
async def tenants(container):
    a = await make_workspace(container)
    b = await make_workspace(container)
    await ingest(container, a, "payroll.md", SECRET_A)
    await ingest(container, b, "payroll.md", SECRET_B)  # same filename/external id in another tenant
    return a, b


async def _chunk_tenants(container, ids):
    from sqlalchemy import select

    from app.models import Chunk

    async with container.session_factory() as s:
        rows = await s.execute(select(Chunk.tenant_id).where(Chunk.id.in_(list(ids))))
        return {r[0] for r in rows}


async def test_dense_and_sparse_never_cross_tenants(container, tenants):
    a, b = tenants
    async with container.session_factory() as s:
        emb = await container.embedding_configs.get_default(s)
    embedder = container.embedders.get(spec_of(emb))
    q = "secret payroll bonus code"
    vec = await embedder.embed_query(q)
    for ws in (a, b):
        scope = RetrievalScope(ws.tenant_id, ws.kb_id)
        dense = await container.vector_store.search(emb.collection_name, vec, scope, top_k=100)
        sparse = await PostgresBM25Search(container.session_factory).search(
            RetrievalQuery(q, scope, top_k=100)
        )
        assert dense and sparse
        assert await _chunk_tenants(container, [h.id for h in dense]) == {ws.tenant_id}
        assert await _chunk_tenants(container, [r.chunk_id for r in sparse]) == {ws.tenant_id}
        assert {h.payload["tenant_id"] for h in dense} == {str(ws.tenant_id)}


async def test_orchestrator_answers_only_from_own_tenant(container, tenants):
    a, _b = tenants
    opts = QueryOptions.from_settings(container.settings.retrieval)
    ra = await container.orchestrator.query(
        tenant_id=a.tenant_id,
        knowledge_base_id=a.kb_id,
        query="What is the payroll bonus code?",
        filters={},
        options=opts,
    )
    assert "ALPHA-7781" in ra.answer and "BRAVO" not in ra.answer
    assert all(c.document_id for c in ra.citations)
    assert all("ALPHA" in c.snippet for c in ra.citations)


async def test_cannot_query_other_tenants_kb(container, tenants):
    a, b = tenants
    from app.core.errors import NotFoundError

    with pytest.raises(NotFoundError):
        await container.orchestrator.query(
            tenant_id=a.tenant_id,
            knowledge_base_id=b.kb_id,
            query="payroll",
            filters={},
            options=QueryOptions.from_settings(container.settings.retrieval),
        )


async def test_api_isolation(container, tenants):
    a, b = tenants
    app = create_app(container.settings, container=container)
    app.state.container = container
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        ha, hb = {"X-API-Key": a.api_key}, {"X-API-Key": b.api_key}

        # Querying B's KB with A's key -> 404 (existence not leaked)
        r = await client.post(
            "/v1/rag/query", headers=ha, json={"knowledge_base_id": str(b.kb_id), "query": "code"}
        )
        assert r.status_code == 404

        # Tenant filters cannot be smuggled in
        r = await client.post(
            "/v1/rag/query",
            headers=ha,
            json={
                "knowledge_base_id": str(a.kb_id),
                "query": "code",
                "filters": {"tenant_id": str(b.tenant_id)},
            },
        )
        assert r.status_code == 422

        # B's trace is invisible to A
        rb = await client.post(
            "/v1/rag/query", headers=hb, json={"knowledge_base_id": str(b.kb_id), "query": "code"}
        )
        assert rb.status_code == 200
        assert (await client.get(f"/v1/traces/{rb.json()['trace_id']}", headers=ha)).status_code == 404
        assert (await client.get(f"/v1/traces/{rb.json()['trace_id']}", headers=hb)).status_code == 200
        assert all("BRAVO" in c["snippet"] for c in rb.json()["citations"])

        # B's KB and documents are invisible to A; uploads into B's KB are rejected
        assert (await client.get(f"/v1/knowledge-bases/{b.kb_id}", headers=ha)).status_code == 404
        r = await client.post(
            "/v1/documents", headers=ha, data={"knowledge_base_id": str(b.kb_id), "text": "x" * 50}
        )
        assert r.status_code == 404

        # No / bad key
        assert (
            await client.post("/v1/rag/query", json={"knowledge_base_id": str(a.kb_id), "query": "x"})
        ).status_code == 401
        bad = {"X-API-Key": "rag_wrong"}
        assert (await client.get(f"/v1/knowledge-bases/{a.kb_id}", headers=bad)).status_code == 401

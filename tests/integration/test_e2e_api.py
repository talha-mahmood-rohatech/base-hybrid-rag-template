"""End-to-end over HTTP: tenant -> KB -> upload (all formats) -> ingestion -> hybrid query -> trace."""

import json

import httpx
import pytest

from app.main import create_app
from scripts.sample_docs import sample_files

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
async def client(container):
    app = create_app(container.settings, container=container)
    app.state.container = container
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_end_to_end(container, client):
    admin = {"X-Admin-Key": container.settings.security.admin_api_key.get_secret_value()}
    assert (await client.post("/v1/admin/tenants", json={"name": "x", "slug": "y"})).status_code == 401
    r = await client.post(
        "/v1/admin/tenants", headers=admin, json={"name": "Northwind", "slug": "northwind-e2e"}
    )
    assert r.status_code == 201, r.text
    h = {"X-API-Key": r.json()["api_key"]}

    r = await client.post(
        "/v1/knowledge-bases", headers=h, json={"name": "policies", "description": "Company policies"}
    )
    assert r.status_code == 201, r.text
    kb = r.json()
    assert kb["embedding_config"]["dimension"] == 1024
    kb_id = kb["id"]

    jobs = []
    for name, data in sample_files().items():
        r = await client.post(
            "/v1/documents",
            headers=h,
            data={"knowledge_base_id": kb_id, "metadata": json.dumps({"origin": "samples"})},
            files={"file": (name, data)},
        )
        assert r.status_code == 202, r.text
        assert r.json()["outcome"] == "created"
        jobs.append(r.json()["job"]["id"])

    # idempotent upload
    r = await client.post(
        "/v1/documents",
        headers=h,
        data={"knowledge_base_id": kb_id},
        files={"file": ("refund_policy.md", sample_files()["refund_policy.md"])},
    )
    assert r.json()["outcome"] == "unchanged" and r.json()["job"] is None

    assert await container.worker.drain() == len(jobs)
    for job_id in jobs:
        j = (await client.get(f"/v1/ingestion/jobs/{job_id}", headers=h)).json()
        assert j["status"] == "succeeded", j

    r = await client.post(
        "/v1/rag/query",
        headers=h,
        json={
            "knowledge_base_id": kb_id,
            "query": "What is the monthly uptime commitment in the SLA?",
            "top_k": 8,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "99.95%" in body["answer"]
    assert body["citations"], body
    for c in body["citations"]:
        assert set(c) >= {"document_id", "document_name", "chunk_id", "page", "section"}
    assert any(c["document_name"] == "sla_agreement.pdf" and c["page"] for c in body["citations"])
    retrieval = body["retrieval"]
    assert retrieval["dense_candidates"] > 0 and retrieval["sparse_candidates"] > 0
    assert (
        retrieval["rrf_candidates"] >= max(retrieval["dense_candidates"], retrieval["sparse_candidates"])
        or retrieval["rrf_candidates"] == 40
    )
    assert retrieval["reranked_candidates"] <= 8
    assert body["results"][0]["reranker_score"] is not None

    # trace explains the selection
    t = (await client.get(f"/v1/traces/{body['trace_id']}", headers=h)).json()
    assert t["query"].startswith("What is the monthly uptime")
    assert t["dense_results"] and t["sparse_results"] and t["fused_results"] and t["reranked_results"]
    assert {"rank", "score", "chunk_id"} <= set(t["dense_results"][0])
    sel = t["selected_chunks"][0]
    assert {"dense_rank", "sparse_rank", "rrf_score", "reranker_score"} <= set(sel["why"])
    assert t["config"]["rrf_k"] == 60 and t["config"]["dense_top_k"] == 50
    assert t["config"]["embedding"]["dimension"] == 1024
    assert t["llm"]["provider"] == "extractive" and t["context_tokens"] > 0
    assert t["citations"] == body["citations"]
    assert t["timings_ms"]["total_ms"] > 0

    # metadata filter narrows to one document type
    r = await client.post(
        "/v1/rag/query",
        headers=h,
        json={
            "knowledge_base_id": kb_id,
            "query": "refund",
            "filters": {"document_type": "md"},
            "generate_answer": False,
        },
    )
    body = r.json()
    assert body["answer"] is None and body["results"]
    assert {x["document_name"] for x in body["results"]} == {"refund_policy.md"}

    # retrieval knobs can be overridden per request
    r = await client.post(
        "/v1/rag/query",
        headers=h,
        json={
            "knowledge_base_id": kb_id,
            "query": "security incident reporting",
            "top_k": 3,
            "options": {"dense_top_k": 5, "sparse_top_k": 5, "rrf_k": 10, "rrf_top_k": 6},
        },
    )
    body = r.json()
    assert body["retrieval"]["dense_candidates"] <= 5 and body["retrieval"]["rrf_candidates"] <= 6
    assert len(body["results"]) <= 3
    assert "24 hours" in body["answer"]


async def test_unknown_answer_has_no_citations(container, client):
    admin = {"X-Admin-Key": container.settings.security.admin_api_key.get_secret_value()}
    key = (
        await client.post("/v1/admin/tenants", headers=admin, json={"name": "e", "slug": "empty-e2e"})
    ).json()["api_key"]
    h = {"X-API-Key": key}
    kb_id = (await client.post("/v1/knowledge-bases", headers=h, json={"name": "empty"})).json()["id"]
    r = await client.post("/v1/rag/query", headers=h, json={"knowledge_base_id": kb_id, "query": "anything?"})
    assert r.status_code == 200
    assert r.json()["citations"] == [] and r.json()["answer"].startswith("I don't know")


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json()["checks"] == {"postgres": "ok", "qdrant": "ok"}

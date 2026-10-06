"""Live end-to-end demonstration against a running stack (docker compose up).

    python scripts/demo.py --api http://localhost:8000 --admin-key change-me-admin-key

Shows: ingestion of PDF/DOCX/MD/HTML/TXT -> embeddings in Qdrant + metadata in PostgreSQL ->
dense + sparse retrieval -> RRF -> cross-encoder reranking -> context -> LLM answer with
citations -> trace -> tenant isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def hr(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def wait_healthy(c: httpx.Client, timeout: float = 1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = c.get("/health")
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(3)
    sys.exit("API did not become healthy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--qdrant", default="http://localhost:6333")
    ap.add_argument("--admin-key", default="change-me-admin-key")
    ap.add_argument("--samples", default=str(ROOT / "samples"))
    ap.add_argument("--query", default="What is the refund policy for annual subscriptions?")
    args = ap.parse_args()

    c = httpx.Client(base_url=args.api, timeout=600)
    health = wait_healthy(c)
    hr("1. Platform health / configuration")
    print(json.dumps(health, indent=2))

    run = uuid.uuid4().hex[:6]
    admin = {"X-Admin-Key": args.admin_key}
    a = c.post(
        "/v1/admin/tenants", headers=admin, json={"name": "Northwind", "slug": f"northwind-{run}"}
    ).json()
    b = c.post("/v1/admin/tenants", headers=admin, json={"name": "Globex", "slug": f"globex-{run}"}).json()
    ha, hb = {"X-API-Key": a["api_key"]}, {"X-API-Key": b["api_key"]}

    hr("2. Create knowledge base (tenant Northwind)")
    kb = c.post(
        "/v1/knowledge-bases", headers=ha, json={"name": "policies", "description": "Company policies"}
    ).json()
    print(json.dumps({k: kb[k] for k in ("id", "name", "fts_language", "embedding_config")}, indent=2))

    hr("3. Upload documents (PDF, DOCX, Markdown, HTML, TXT)")
    jobs = {}
    for f in sorted(Path(args.samples).iterdir()):
        r = c.post(
            "/v1/documents",
            headers=ha,
            data={
                "knowledge_base_id": kb["id"],
                "metadata": json.dumps({"origin": "demo"}),
                "source": "samples",
            },
            files={"file": (f.name, f.read_bytes())},
        )
        r.raise_for_status()
        body = r.json()
        print(
            f"  {body['outcome']:<10} {f.name:<26} version={body['version']['version_number']} "
            f"checksum={body['version']['checksum'][:12]}..."
        )
        if body["job"]:
            jobs[f.name] = body["job"]["id"]
    again = c.post(
        "/v1/documents",
        headers=ha,
        data={"knowledge_base_id": kb["id"]},
        files={"file": ("refund_policy.md", (Path(args.samples) / "refund_policy.md").read_bytes())},
    ).json()
    print(f"  re-upload of identical refund_policy.md -> outcome={again['outcome']} (idempotent, no new job)")

    hr("4. Ingestion jobs (parse -> clean -> chunk -> embed -> Qdrant + PostgreSQL)")
    t0 = time.time()
    pending = dict(jobs)
    while pending:
        for name, jid in list(pending.items()):
            j = c.get(f"/v1/ingestion/jobs/{jid}", headers=ha).json()
            if j["status"] in ("succeeded", "failed"):
                s = j["stats"]
                print(
                    f"  {j['status']:<10} {name:<26} chunks={s.get('chunks')} tokens={s.get('tokens')} "
                    f"embed={s.get('timings_ms', {}).get('embed_ms')}ms collection={s.get('collection')}"
                    + (f" ERROR={j['error']}" if j["error"] else "")
                )
                pending.pop(name)
        if pending:
            time.sleep(2)
    print(f"  all jobs finished in {time.time() - t0:.1f}s")

    collection = kb["embedding_config"]["collection_name"]
    hr("5. Verify vectors in Qdrant (filtered by tenant + knowledge base)")
    q = httpx.post(
        f"{args.qdrant}/collections/{collection}/points/count",
        json={
            "exact": True,
            "filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": a["id"]}},
                    {"key": "knowledge_base_id", "match": {"value": kb["id"]}},
                ]
            },
        },
    ).json()
    info = httpx.get(f"{args.qdrant}/collections/{collection}").json()["result"]
    print(f"  collection={collection}")
    print(
        f"  vector size={info['config']['params']['vectors']['size']} distance={info['config']['params']['vectors']['distance']}"
    )
    print(f"  points for this tenant/KB: {q['result']['count']}")
    print(f"  payload indexes: {sorted(info.get('payload_schema', {}))}")

    hr(f"6. RAG query: {args.query!r}")
    t0 = time.time()
    r = c.post(
        "/v1/rag/query", headers=ha, json={"knowledge_base_id": kb["id"], "query": args.query, "top_k": 8}
    )
    r.raise_for_status()
    res = r.json()
    print(f"  latency: {time.time() - t0:.1f}s   retrieval: {res['retrieval']}")
    print("\n  Top reranked chunks:")
    print(f"  {'#':<3}{'document':<26}{'section/page':<46}{'dense':>6}{'sparse':>7}{'rrf':>9}{'rerank':>9}")
    for x in res["results"]:
        where = (x["section"] or f"page {x['page']}")[:44]
        print(
            f"  {x['rank']:<3}{x['document_name']:<26}{where:<46}{x['dense_rank'] or '-':>6}"
            f"{x['sparse_rank'] or '-':>7}{x['rrf_score']:>9.5f}{x['reranker_score'] if x['reranker_score'] is not None else float('nan'):>9.3f}"
        )
    print(f"\n  ANSWER ({res['usage']['model'] if res['usage'] else '-'}):\n  {res['answer']}\n")
    print("  CITATIONS:")
    for ci in res["citations"]:
        print(
            f"   [{ci['citation_id']}] {ci['document_name']} | section={ci['section']} | page={ci['page']} "
            f"| chunk={ci['chunk_id'][:8]} | rerank={ci['reranker_score']}"
        )

    hr("7. Trace: why each chunk was selected")
    t = c.get(f"/v1/traces/{res['trace_id']}", headers=ha).json()
    print(f"  trace_id={t['id']} status={t['status']} context_tokens={t['context_tokens']}")
    print(
        f"  config: rrf_k={t['config']['rrf_k']} dense_top_k={t['config']['dense_top_k']} "
        f"sparse_top_k={t['config']['sparse_top_k']} rrf_top_k={t['config']['rrf_top_k']}"
    )
    print(f"  reranker={t['config']['reranker']}  llm={t['llm'].get('model')} usage={t['llm'].get('usage')}")
    print(f"  timings_ms={ {k: v for k, v in t['timings_ms'].items() if k != 'counts'} }")
    for s in t["selected_chunks"][:4]:
        print(f"   [{s['citation_id']}] {s['document_name']}: {s['why']}")

    hr("8. Tenant isolation")
    kb_b = c.post("/v1/knowledge-bases", headers=hb, json={"name": "policies"}).json()
    c.post(
        "/v1/documents",
        headers=hb,
        data={
            "knowledge_base_id": kb_b["id"],
            "filename": "refunds.md",
            "text": "# Globex refunds\n\nGlobex never issues refunds. Code GLX-SECRET-99.",
        },
    )
    time.sleep(1)
    for _ in range(120):
        d = c.post(
            "/v1/rag/query",
            headers=hb,
            json={"knowledge_base_id": kb_b["id"], "query": "refunds", "generate_answer": False},
        ).json()
        if d["results"]:
            break
        time.sleep(2)
    cross = c.post("/v1/rag/query", headers=ha, json={"knowledge_base_id": kb_b["id"], "query": "refunds"})
    print(f"  Northwind querying Globex's KB -> HTTP {cross.status_code} ({cross.json()['error']['code']})")
    own = c.post(
        "/v1/rag/query",
        headers=ha,
        json={
            "knowledge_base_id": kb["id"],
            "query": "GLX-SECRET-99 Globex refunds",
            "generate_answer": False,
        },
    ).json()
    leaked = [x for x in own["results"] if "GLX-SECRET" in x["text"]]
    print(
        f"  Northwind searching its own KB for Globex's secret -> {len(own['results'])} results, leaked={len(leaked)}"
    )
    smuggle = c.post(
        "/v1/rag/query",
        headers=ha,
        json={"knowledge_base_id": kb["id"], "query": "x", "filters": {"tenant_id": b["id"]}},
    )
    print(f"  Filter on tenant_id -> HTTP {smuggle.status_code} (rejected)")
    trace_b = c.get(f"/v1/traces/{d.get('trace_id')}", headers=ha)
    print(f"  Northwind reading Globex's trace -> HTTP {trace_b.status_code}")
    ok = cross.status_code == 404 and not leaked and smuggle.status_code == 422 and trace_b.status_code == 404
    print(f"\n  TENANT ISOLATION: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

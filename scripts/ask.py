"""Ask a knowledge base a question from the terminal.

python scripts/ask.py --tenant-slug ztbl --kb islamic-banking "What is Ijarah?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

SECRETS = Path(__file__).resolve().parents[1] / ".secrets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--tenant-slug", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()

    key = json.loads((SECRETS / f"{args.tenant_slug}.json").read_text())["api_key"]
    c = httpx.Client(base_url=args.api, headers={"X-API-Key": key}, timeout=600)
    kb = next((k for k in c.get("/v1/knowledge-bases").json() if k["name"] == args.kb), None)
    if kb is None:
        sys.exit(f"knowledge base '{args.kb}' not found")
    r = c.post(
        "/v1/rag/query",
        json={"knowledge_base_id": kb["id"], "query": args.question, "top_k": args.top_k},
    )
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text}")
    res = r.json()
    print(f"\n{res['answer']}\n")
    for ci in res["citations"]:
        where = ci["section"] or (f"page {ci['page']}" if ci["page"] else "")
        print(f"  [{ci['citation_id']}] {ci['document_name']} | {where}")
    usage = res.get("usage") or {}
    print(
        f"\n  model={usage.get('model')} tokens={usage.get('prompt_tokens')}+{usage.get('completion_tokens')} "
        f"latency={res['latency_ms'] / 1000:.1f}s retrieval={res['retrieval']} trace_id={res['trace_id']}"
    )
    if args.show_chunks:
        for x in res["results"]:
            print(
                f"\n  #{x['rank']} dense={x['dense_rank']} sparse={x['sparse_rank']} "
                f"rrf={x['rrf_score']:.4f} rerank={x['reranker_score']}\n  {x['section']}\n  {x['text'][:300]}"
            )


if __name__ == "__main__":
    main()

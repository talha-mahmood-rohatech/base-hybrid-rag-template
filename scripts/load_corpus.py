"""Load a folder of documents into a knowledge base over the HTTP API (works for any corpus).

    python scripts/load_corpus.py --dir corpora/ztbl --tenant-slug ztbl --tenant-name "ZTBL" \
        --kb islamic-banking --api http://localhost:8000 --admin-key <admin key>

Creates the tenant (once) and the knowledge base (once), uploads every supported file
(re-runs are idempotent: unchanged files are skipped, changed files become new versions),
waits for ingestion and stores the tenant credentials in .secrets/<slug>.json (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm", ".txt"}
SECRETS = Path(__file__).resolve().parents[1] / ".secrets"


def tenant_key(c: httpx.Client, args) -> str:
    path = SECRETS / f"{args.tenant_slug}.json"
    if args.api_key:
        return args.api_key
    if path.exists():
        return json.loads(path.read_text())["api_key"]
    r = c.post(
        "/v1/admin/tenants",
        headers={"X-Admin-Key": args.admin_key},
        json={"name": args.tenant_name or args.tenant_slug, "slug": args.tenant_slug},
    )
    if r.status_code == 409:
        sys.exit(f"Tenant '{args.tenant_slug}' exists but its key is not in {path}; pass --api-key")
    r.raise_for_status()
    SECRETS.mkdir(exist_ok=True)
    path.write_text(json.dumps({"tenant_id": r.json()["id"], "api_key": r.json()["api_key"]}, indent=2))
    print(f"created tenant '{args.tenant_slug}', credentials saved to {path}")
    return r.json()["api_key"]


def knowledge_base(c: httpx.Client, h: dict, args) -> dict:
    for kb in c.get("/v1/knowledge-bases", headers=h).json():
        if kb["name"] == args.kb:
            return kb
    body: dict = {"name": args.kb, "fts_language": args.language}
    if args.chunk_size:
        body["chunking"] = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}
    r = c.post("/v1/knowledge-bases", headers=h, json=body)
    r.raise_for_status()
    print(f"created knowledge base '{args.kb}'")
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tenant-slug", required=True)
    ap.add_argument("--tenant-name")
    ap.add_argument("--kb", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--admin-key", default="change-me-admin-key")
    ap.add_argument("--api-key", help="existing tenant key (skips tenant creation)")
    ap.add_argument("--language", default="english")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--chunk-overlap", type=int, default=72)
    ap.add_argument("--source", help="value stored in the 'source' field of every document")
    args = ap.parse_args()

    c = httpx.Client(base_url=args.api, timeout=600)
    h = {"X-API-Key": tenant_key(c, args)}
    kb = knowledge_base(c, h, args)

    jobs = {}
    for f in sorted(Path(args.dir).iterdir()):
        if f.suffix.lower() not in SUPPORTED:
            continue
        data = {"knowledge_base_id": kb["id"], "external_id": f.stem}
        if args.source:
            data["source"] = args.source
        r = c.post("/v1/documents", headers=h, data=data, files={"file": (f.name, f.read_bytes())})
        r.raise_for_status()
        body = r.json()
        print(f"  {body['outcome']:<12} {f.name} (version {body['version']['version_number']})")
        if body["job"]:
            jobs[f.name] = body["job"]["id"]

    while jobs:
        for name, jid in list(jobs.items()):
            j = c.get(f"/v1/ingestion/jobs/{jid}", headers=h).json()
            if j["status"] in ("succeeded", "failed"):
                detail = f"chunks={j['stats'].get('chunks')}" if j["status"] == "succeeded" else j["error"]
                print(f"  {j['status']:<12} {name}: {detail}")
                del jobs[name]
        if jobs:
            time.sleep(2)
    print(f"\nknowledge_base_id={kb['id']}")


if __name__ == "__main__":
    main()

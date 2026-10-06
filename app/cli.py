"""Operator CLI.

python -m app.cli create-tenant --name "Acme" --slug acme
python -m app.cli ingest-dir --tenant acme --kb policies samples/
python -m app.cli query --tenant acme --kb policies "What is the refund policy?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

from app.container import Container
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.models import KnowledgeBase, Tenant
from app.rag.orchestrator import QueryOptions
from app.services.tenants import create_tenant


async def _tenant(c: Container, slug: str) -> Tenant:
    async with c.session_factory() as s:
        t = (await s.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
    if t is None:
        sys.exit(f"tenant '{slug}' not found")
    return t


async def _kb(c: Container, tenant: Tenant, name: str, create: bool = False) -> KnowledgeBase:
    async with c.session_factory() as s, s.begin():
        kb = (
            await s.execute(
                select(KnowledgeBase).where(KnowledgeBase.tenant_id == tenant.id, KnowledgeBase.name == name)
            )
        ).scalar_one_or_none()
        if kb is None:
            if not create:
                sys.exit(f"knowledge base '{name}' not found")
            emb = await c.embedding_configs.get_default(s)
            kb = KnowledgeBase(
                id=uuid.uuid4(),
                tenant_id=tenant.id,
                name=name,
                embedding_config_id=emb.id,
                fts_language=c.settings.retrieval.fts_language,
                settings={},
            )
            s.add(kb)
    return kb


async def main_async(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="rag-cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    ct = sub.add_parser("create-tenant")
    ct.add_argument("--name", required=True)
    ct.add_argument("--slug", required=True)
    ing = sub.add_parser("ingest-dir")
    ing.add_argument("--tenant", required=True)
    ing.add_argument("--kb", required=True)
    ing.add_argument("path")
    q = sub.add_parser("query")
    q.add_argument("--tenant", required=True)
    q.add_argument("--kb", required=True)
    q.add_argument("--no-answer", action="store_true")
    q.add_argument("text")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")
    c = await Container.create(settings)
    try:
        if args.cmd == "create-tenant":
            async with c.session_factory() as s, s.begin():
                tenant, key = await create_tenant(s, args.name, args.slug)
            print(json.dumps({"tenant_id": str(tenant.id), "api_key": key}, indent=2))
        elif args.cmd == "ingest-dir":
            tenant = await _tenant(c, args.tenant)
            kb = await _kb(c, tenant, args.kb, create=True)
            files = await asyncio.to_thread(
                lambda: sorted(f for f in Path(args.path).iterdir() if f.is_file())
            )
            for f in files:
                async with c.session_factory() as s, s.begin():
                    res = await c.documents.upload(
                        s,
                        tenant_id=tenant.id,
                        knowledge_base_id=kb.id,
                        data=await asyncio.to_thread(f.read_bytes),
                        filename=f.name,
                    )
                print(f"{res.outcome:12s} {f.name} (version {res.version.version_number})")
            n = await c.worker.drain()
            print(f"processed {n} ingestion job(s)")
        elif args.cmd == "query":
            tenant = await _tenant(c, args.tenant)
            kb = await _kb(c, tenant, args.kb)
            answer = await c.orchestrator.query(
                tenant_id=tenant.id,
                knowledge_base_id=kb.id,
                query=args.text,
                filters={},
                options=QueryOptions.from_settings(settings.retrieval, generate_answer=not args.no_answer),
            )
            print(f"\nANSWER:\n{answer.answer}\n")
            for ci in answer.citations:
                print(f"  [{ci.citation_id}] {ci.document_name} | page={ci.page} | section={ci.section}")
            print(f"\nretrieval: {answer.counts}   trace_id: {answer.trace_id}")
    finally:
        await c.aclose()


def main() -> None:
    asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    main()

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.container import Container
from app.models import KnowledgeBase
from app.services.tenants import create_tenant


@dataclass
class Workspace:
    tenant_id: uuid.UUID
    api_key: str
    kb_id: uuid.UUID


async def make_workspace(
    c: Container, *, name: str | None = None, fts_language: str = "english"
) -> Workspace:
    slug = name or f"t-{uuid.uuid4().hex[:10]}"
    async with c.session_factory() as s, s.begin():
        tenant, key = await create_tenant(s, slug, slug)
        emb = await c.embedding_configs.get_default(s)
        kb = KnowledgeBase(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            name="kb",
            embedding_config_id=emb.id,
            fts_language=fts_language,
            settings={},
        )
        s.add(kb)
    return Workspace(tenant.id, key, kb.id)


async def ingest(c: Container, ws: Workspace, filename: str, data: bytes | str, **kw):
    if isinstance(data, str):
        data = data.encode()
    async with c.session_factory() as s, s.begin():
        res = await c.documents.upload(
            s, tenant_id=ws.tenant_id, knowledge_base_id=ws.kb_id, data=data, filename=filename, **kw
        )
    await c.worker.drain()
    return res

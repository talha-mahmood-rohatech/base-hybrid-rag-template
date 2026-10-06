from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_container, get_session, get_tenant
from app.container import Container
from app.core.errors import ConflictError, ValidationError
from app.ingestion.embedding_configs import collection_of
from app.ingestion.service import get_kb
from app.models import KnowledgeBase
from app.schemas.api import KnowledgeBaseCreate, KnowledgeBaseOut

router = APIRouter(prefix="/v1/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseOut, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> KnowledgeBase:
    known = await session.execute(
        text("SELECT 1 FROM pg_ts_config WHERE cfgname = :l"), {"l": body.fts_language}
    )
    if known.first() is None:
        raise ValidationError(f"Unknown full-text search language '{body.fts_language}'")

    if body.embedding_config_id:
        emb = await container.embedding_configs.get(session, body.embedding_config_id)
    else:
        emb = await container.embedding_configs.get_default(session)

    settings = {}
    if body.chunking:
        chunking = body.chunking.model_dump(exclude_none=True)
        merged = container.settings.chunking.model_copy(update=chunking)
        if merged.chunk_overlap >= merged.chunk_size:
            raise ValidationError("chunk_overlap must be smaller than chunk_size")
        settings["chunking"] = chunking

    kb = KnowledgeBase(
        id=uuid.uuid4(),
        tenant_id=tenant.tenant_id,
        name=body.name,
        description=body.description,
        embedding_config_id=emb.id,
        embedding_config=emb,
        fts_language=body.fts_language,
        settings=settings,
    )
    session.add(kb)
    try:
        await session.commit()
    except IntegrityError as exc:
        raise ConflictError(f"Knowledge base '{body.name}' already exists") from exc
    await container.vector_store.ensure_collection(collection_of(emb))
    return await get_kb(session, tenant.tenant_id, kb.id)


@router.get("", response_model=list[KnowledgeBaseOut])
async def list_knowledge_bases(
    tenant: TenantContext = Depends(get_tenant), session: AsyncSession = Depends(get_session)
) -> list[KnowledgeBase]:
    rows = await session.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.tenant_id == tenant.tenant_id)
        .order_by(KnowledgeBase.created_at)
    )
    return list(rows.scalars().all())


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
async def get_knowledge_base(
    kb_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeBase:
    return await get_kb(session, tenant.tenant_id, kb_id)

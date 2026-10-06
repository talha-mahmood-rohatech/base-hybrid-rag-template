"""Chunk hydration from PostgreSQL (source of truth for text + active state)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.retrieval.types import ChunkView, RetrievalScope


class ChunkRepository:
    async def get_many(
        self, session: AsyncSession, scope: RetrievalScope, ids: Sequence[uuid.UUID]
    ) -> dict[uuid.UUID, ChunkView]:
        if not ids:
            return {}
        stmt = (
            select(Chunk, Document.name)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.id.in_(list(ids)),
                Chunk.tenant_id == scope.tenant_id,
                Chunk.knowledge_base_id == scope.knowledge_base_id,
            )
        )
        if scope.active_only:
            stmt = stmt.where(Chunk.is_active.is_(True))
        rows = (await session.execute(stmt)).all()
        out: dict[uuid.UUID, ChunkView] = {}
        for chunk, doc_name in rows:
            out[chunk.id] = ChunkView(
                chunk_id=chunk.id,
                tenant_id=chunk.tenant_id,
                knowledge_base_id=chunk.knowledge_base_id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                document_name=doc_name,
                document_type=chunk.document_type,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=chunk.token_count,
                page=chunk.page,
                page_end=chunk.page_end,
                section=chunk.section,
                checksum=chunk.checksum,
                metadata=chunk.metadata_ or {},
            )
        return out

"""Ingestion pipeline: Load -> Parse -> Clean -> Chunk -> Embed -> Qdrant + PostgreSQL -> Activate.

Idempotency: chunk ids are deterministic (uuid5(version_id, chunk_index)), and any partial
output of a previous attempt for the same version is deleted first, so a job can be retried
safely. Activation happens in a single PostgreSQL transaction (source of truth); Qdrant
``is_active`` flags are flipped afterwards and retrieval re-checks PostgreSQL on hydration.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import ChunkingSettings
from app.core.tokens import TokenCounter
from app.core.utils import deterministic_chunk_id, sha256_hex
from app.ingestion.blobstore import BlobStore
from app.ingestion.chunking.base import ChunkDraft
from app.ingestion.chunking.recursive import RecursiveStructureChunker
from app.ingestion.cleaning import clean_text
from app.ingestion.embedding_configs import collection_of, spec_of
from app.ingestion.loaders.base import ParsedDocument, Segment
from app.ingestion.loaders.registry import LoaderRegistry
from app.models import Chunk, Document, DocumentVersion, EmbeddingConfig, KnowledgeBase, VersionStatus
from app.providers.registry import EmbeddingProviderCache
from app.providers.vectorstores.base import VectorPoint, VectorStore

logger = logging.getLogger(__name__)

EMBED_BATCH = 64


@dataclass(slots=True)
class IngestionResult:
    document_version_id: uuid.UUID
    chunk_count: int
    activated: bool
    timings_ms: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def embedding_text(title: str | None, section: str | None, body: str) -> str:
    header = "\n".join(p for p in (title, section) if p)
    return f"{header}\n\n{body}" if header else body


class IngestionPipeline:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        blob_store: BlobStore,
        vector_store: VectorStore,
        embedders: EmbeddingProviderCache,
        loaders: LoaderRegistry,
        token_counter: TokenCounter,
        chunking: ChunkingSettings,
    ) -> None:
        self.session_factory = session_factory
        self.blob_store = blob_store
        self.vector_store = vector_store
        self.embedders = embedders
        self.loaders = loaders
        self.token_counter = token_counter
        self.chunking = chunking

    def chunker_for(self, kb: KnowledgeBase) -> RecursiveStructureChunker:
        cfg = self.chunking.model_copy(update=(kb.settings or {}).get("chunking", {}))
        cfg = ChunkingSettings.model_validate(cfg.model_dump())
        return RecursiveStructureChunker(
            self.token_counter,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            unit=cfg.unit,
            min_chunk_chars=cfg.min_chunk_chars,
        )

    def parse(self, data: bytes, document_type: str, filename: str | None) -> ParsedDocument:
        parsed = self.loaders.get(document_type).load(data, filename)
        segments = []
        for seg in parsed.segments:
            cleaned = clean_text(seg.text, fix_hyphenation=document_type == "pdf")
            if cleaned:
                segments.append(Segment(text=cleaned, page=seg.page, section=seg.section))
        parsed.segments = segments
        return parsed

    async def run(self, version_id: uuid.UUID) -> IngestionResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = round((now - t) * 1000, 2)
            t = now

        async with self.session_factory() as session:
            version = await session.get(DocumentVersion, version_id)
            if version is None:
                raise ValueError(f"Document version {version_id} not found")
            document = await session.get(Document, version.document_id)
            kb = await session.get(KnowledgeBase, version.knowledge_base_id)
            assert document is not None and kb is not None
            emb_cfg = await session.get(EmbeddingConfig, kb.embedding_config_id)
            assert emb_cfg is not None
            version.status = VersionStatus.processing
            version.error = None
            await session.commit()

        tenant_id, kb_id, doc_id = version.tenant_id, version.knowledge_base_id, document.id
        data = await self.blob_store.get(version.blob_uri)
        lap("load_ms")

        parsed = self.parse(data, document.document_type, document.name)
        lap("parse_clean_ms")

        drafts = self.chunker_for(kb).chunk(parsed)
        if not drafts:
            raise ValueError("Document produced no chunks (empty or unparseable content)")
        lap("chunk_ms")

        title = parsed.title or PurePath(document.name).stem
        embedder = self.embedders.get(spec_of(emb_cfg), emb_cfg.params)
        texts = [embedding_text(title, d.section, d.text) for d in drafts]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            vectors.extend(await embedder.embed_documents(texts[i : i + EMBED_BATCH]))
        lap("embed_ms")

        collection = collection_of(emb_cfg)
        await self.vector_store.ensure_collection(collection)
        # Clean any partial output from a previous attempt (idempotent retries).
        await self.vector_store.delete_by_document_version(collection.name, tenant_id, version_id)
        permissions = list(document.permissions or ["public"])
        points = [
            VectorPoint(
                id=deterministic_chunk_id(version_id, d.index),
                vector=vec,
                payload=self._payload(d, document, version, emb_cfg, permissions),
            )
            for d, vec in zip(drafts, vectors, strict=True)
        ]
        await self.vector_store.upsert(collection, points)
        lap("qdrant_upsert_ms")

        async with self.session_factory() as session:
            await session.execute(delete(Chunk).where(Chunk.document_version_id == version_id))
            rows = [
                {
                    "id": deterministic_chunk_id(version_id, d.index),
                    "tenant_id": tenant_id,
                    "knowledge_base_id": kb_id,
                    "document_id": doc_id,
                    "document_version_id": version_id,
                    "embedding_config_id": emb_cfg.id,
                    "chunk_index": d.index,
                    "text": d.text,
                    "token_count": d.token_count,
                    "char_start": d.char_start,
                    "char_end": d.char_end,
                    "page": d.page,
                    "page_end": d.page_end,
                    "section": d.section,
                    "checksum": sha256_hex(d.text),
                    "document_type": document.document_type,
                    "source": document.source,
                    "permissions": permissions,
                    "metadata_": dict(document.metadata_ or {}),
                    "is_active": False,
                }
                for d in drafts
            ]
            await session.execute(insert(Chunk), rows)
            await session.execute(
                text(
                    """
                    UPDATE chunks c SET search_vector =
                        setweight(to_tsvector(CAST(:cfg AS regconfig), coalesce(:title, '') || ' ' || coalesce(c.section, '')), 'A')
                        || setweight(to_tsvector(CAST(:cfg AS regconfig), c.text), 'B')
                    WHERE c.document_version_id = :vid
                    """
                ),
                {"cfg": kb.fts_language, "title": title, "vid": version_id},
            )
            await session.commit()
        lap("postgres_write_ms")

        activated, previous = await self._activate(
            version_id, len(drafts), title, parsed, embedder_name=emb_cfg.model
        )
        if activated:
            await self.vector_store.set_active(collection.name, tenant_id, version_id, True)
            if previous is not None:
                await self.vector_store.set_active(collection.name, tenant_id, previous, False)
        lap("activate_ms")

        return IngestionResult(
            document_version_id=version_id,
            chunk_count=len(drafts),
            activated=activated,
            timings_ms=timings,
            stats={
                "chunks": len(drafts),
                "segments": len(parsed.segments),
                "pages": parsed.page_count,
                "tokens": sum(d.token_count for d in drafts),
                "embedding_model": emb_cfg.model,
                "embedding_dimension": emb_cfg.dimension,
                "collection": collection.name,
                "replaced_version_id": str(previous) if previous else None,
            },
        )

    @staticmethod
    def _payload(
        d: ChunkDraft,
        document: Document,
        version: DocumentVersion,
        emb_cfg: EmbeddingConfig,
        permissions: list[str],
    ) -> dict[str, Any]:
        return {
            "tenant_id": str(document.tenant_id),
            "knowledge_base_id": str(document.knowledge_base_id),
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "chunk_id": str(deterministic_chunk_id(version.id, d.index)),
            "chunk_index": d.index,
            "document_type": document.document_type,
            "source": document.source,
            "page": d.page,
            "section": d.section,
            "permissions": permissions,
            "is_active": False,
            "embedding_version": emb_cfg.version,
            "metadata": dict(document.metadata_ or {}),
        }

    async def _activate(
        self, version_id: uuid.UUID, chunk_count: int, title: str, parsed: ParsedDocument, embedder_name: str
    ) -> tuple[bool, uuid.UUID | None]:
        """Make this version the active one unless a newer version is already active."""
        async with self.session_factory() as session, session.begin():
            version = await session.get(DocumentVersion, version_id)
            assert version is not None
            document = (
                await session.execute(
                    select(Document).where(Document.id == version.document_id).with_for_update()
                )
            ).scalar_one()
            version.status = VersionStatus.ready
            version.chunk_count = chunk_count
            version.title = title
            version.page_count = parsed.page_count
            version.embedding_config_id = (
                await session.execute(
                    select(KnowledgeBase.embedding_config_id).where(
                        KnowledgeBase.id == version.knowledge_base_id
                    )
                )
            ).scalar_one()
            version.metadata_ = {
                **(version.metadata_ or {}),
                "parser": parsed.parser,
                "embedding_model": embedder_name,
            }
            version.ingested_at = datetime.now(UTC)

            previous_id = document.active_version_id
            if previous_id is not None and previous_id != version_id:
                prev = await session.get(DocumentVersion, previous_id)
                if prev is not None and prev.version_number > version.version_number:
                    logger.info(
                        "newer version already active; keeping it", extra={"version": str(version_id)}
                    )
                    return False, None
            if previous_id is not None and previous_id != version_id:
                await session.execute(
                    update(DocumentVersion).where(DocumentVersion.id == previous_id).values(is_active=False)
                )
                await session.execute(
                    update(Chunk).where(Chunk.document_version_id == previous_id).values(is_active=False)
                )
            await session.execute(
                update(Chunk).where(Chunk.document_version_id == version_id).values(is_active=True)
            )
            version.is_active = True
            document.active_version_id = version_id
            return True, (previous_id if previous_id != version_id else None)

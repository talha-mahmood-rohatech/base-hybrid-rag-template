"""Document registration, versioning (checksum-based idempotency) and ingestion job creation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.utils import sha256_hex
from app.ingestion.blobstore import BlobStore
from app.ingestion.loaders.registry import LoaderRegistry
from app.models import (
    Document,
    DocumentVersion,
    IngestionJob,
    JobStatus,
    KnowledgeBase,
    VersionStatus,
)

UploadOutcome = Literal["created", "new_version", "unchanged", "requeued"]


@dataclass(slots=True)
class UploadResult:
    document: Document
    version: DocumentVersion
    job: IngestionJob | None
    outcome: UploadOutcome


async def get_kb(session: AsyncSession, tenant_id: uuid.UUID, kb_id: uuid.UUID) -> KnowledgeBase:
    kb = await session.get(KnowledgeBase, kb_id)
    # Same error for "missing" and "other tenant's" - never leak existence across tenants.
    if kb is None or kb.tenant_id != tenant_id:
        raise NotFoundError(f"Knowledge base {kb_id} not found")
    return kb


class DocumentService:
    def __init__(self, blob_store: BlobStore, loaders: LoaderRegistry, *, max_attempts: int = 3) -> None:
        self.blob_store = blob_store
        self.loaders = loaders
        self.max_attempts = max_attempts

    async def upload(
        self,
        session: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        data: bytes,
        filename: str,
        mime_type: str | None = None,
        external_id: str | None = None,
        document_type: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
        force: bool = False,
        enqueue: bool = True,
    ) -> UploadResult:
        if not data:
            raise ValidationError("Document is empty")
        kb = await get_kb(session, tenant_id, knowledge_base_id)
        doc_type = document_type or self.loaders.detect_type(filename, mime_type, data)
        self.loaders.get(doc_type)  # validates support
        checksum = sha256_hex(data)
        external_id = external_id or filename

        document = (
            await session.execute(
                select(Document)
                .where(Document.knowledge_base_id == kb.id, Document.external_id == external_id)
                .with_for_update()
            )
        ).scalar_one_or_none()

        outcome: UploadOutcome
        if document is None:
            document = Document(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                external_id=external_id,
                name=filename,
                document_type=doc_type,
                source=source,
                mime_type=mime_type,
                metadata_=metadata or {},
                permissions=permissions or ["public"],
                latest_version_number=0,
            )
            session.add(document)
            await session.flush()
            outcome = "created"
        else:
            if document.document_type != doc_type:
                raise ConflictError(
                    f"Document '{external_id}' exists with type '{document.document_type}', got '{doc_type}'"
                )
            # Metadata/permissions follow the latest upload.
            document.name = filename
            if source is not None:
                document.source = source
            if metadata is not None:
                document.metadata_ = metadata
            if permissions is not None:
                document.permissions = permissions
            outcome = "new_version"

            existing = (
                await session.execute(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == document.id, DocumentVersion.checksum == checksum
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                needs_job = force or existing.status == VersionStatus.failed
                job = await self.enqueue(session, existing) if needs_job else None
                return UploadResult(document, existing, job, "requeued" if job else "unchanged")

        document.latest_version_number += 1
        version_id = uuid.uuid4()
        blob_uri = await self.blob_store.put(tenant_id, checksum, data, PurePath(filename).suffix)
        version = DocumentVersion(
            id=version_id,
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            document_id=document.id,
            version_number=document.latest_version_number,
            checksum=checksum,
            size_bytes=len(data),
            blob_uri=blob_uri,
            status=VersionStatus.pending,
            is_active=False,
            metadata_={"filename": filename, "mime_type": mime_type},
        )
        session.add(version)
        await session.flush()
        job = await self.enqueue(session, version) if enqueue else None
        return UploadResult(document, version, job, outcome)

    async def enqueue(self, session: AsyncSession, version: DocumentVersion) -> IngestionJob:
        active = (
            await session.execute(
                select(IngestionJob).where(
                    IngestionJob.document_version_id == version.id,
                    IngestionJob.status.in_([JobStatus.queued, JobStatus.running]),
                )
            )
        ).scalar_one_or_none()
        if active is not None:
            return active
        job = IngestionJob(
            id=uuid.uuid4(),
            tenant_id=version.tenant_id,
            knowledge_base_id=version.knowledge_base_id,
            document_id=version.document_id,
            document_version_id=version.id,
            status=JobStatus.queued,
            attempts=0,
            max_attempts=self.max_attempts,
            stats={},
        )
        if version.status == VersionStatus.failed:
            version.status = VersionStatus.pending
        session.add(job)
        await session.flush()
        return job

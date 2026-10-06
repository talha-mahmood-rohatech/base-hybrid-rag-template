from __future__ import annotations

from collections.abc import Sequence

from app.models import Document, DocumentVersion, IngestionJob
from app.schemas.api import DocumentOut, DocumentVersionOut, IngestionJobOut


def version_out(v: DocumentVersion) -> DocumentVersionOut:
    return DocumentVersionOut(
        id=v.id,
        version_number=v.version_number,
        checksum=v.checksum,
        size_bytes=v.size_bytes,
        status=v.status.value if hasattr(v.status, "value") else str(v.status),
        is_active=v.is_active,
        chunk_count=v.chunk_count,
        page_count=v.page_count,
        title=v.title,
        error=v.error,
        created_at=v.created_at,
        ingested_at=v.ingested_at,
    )


def job_out(j: IngestionJob) -> IngestionJobOut:
    return IngestionJobOut(
        id=j.id,
        document_id=j.document_id,
        document_version_id=j.document_version_id,
        status=j.status.value if hasattr(j.status, "value") else str(j.status),
        attempts=j.attempts,
        max_attempts=j.max_attempts,
        error=j.error,
        stats=j.stats or {},
        created_at=j.created_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


def document_out(d: Document, versions: Sequence[DocumentVersion]) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        knowledge_base_id=d.knowledge_base_id,
        external_id=d.external_id,
        name=d.name,
        document_type=d.document_type,
        source=d.source,
        metadata=d.metadata_ or {},
        permissions=list(d.permissions or []),
        active_version_id=d.active_version_id,
        versions=[version_out(v) for v in versions],
    )

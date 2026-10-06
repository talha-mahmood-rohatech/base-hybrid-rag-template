from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import TenantContext, get_container, get_session, get_tenant
from app.api.serializers import document_out, job_out, version_out
from app.container import Container
from app.core.errors import NotFoundError, ValidationError
from app.ingestion.embedding_configs import collection_of
from app.ingestion.service import get_kb
from app.models import Document, EmbeddingConfig
from app.schemas.api import DocumentOut, DocumentUploadOut

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _parse_json(value: str | None, field: str, expected: type) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"'{field}' must be valid JSON") from exc
    if not isinstance(parsed, expected):
        raise ValidationError(f"'{field}' must be a JSON {expected.__name__}")
    return parsed


@router.post("", response_model=DocumentUploadOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    knowledge_base_id: uuid.UUID = Form(...),
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None, description="Raw text/markdown instead of a file"),
    filename: str | None = Form(default=None),
    external_id: str | None = Form(default=None, description="Stable id; re-uploads create new versions"),
    document_type: str | None = Form(default=None),
    source: str | None = Form(default=None),
    metadata: str | None = Form(default=None, description="JSON object"),
    permissions: str | None = Form(default=None, description='JSON list, e.g. ["public"]'),
    force: bool = Form(default=False, description="Re-ingest even if content is unchanged"),
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> DocumentUploadOut:
    """Register a document (or a new version of it) and enqueue its ingestion job."""
    max_bytes = container.settings.storage.max_upload_bytes
    if file is not None:
        data = await file.read(max_bytes + 1)
        name = filename or file.filename or "upload"
        mime = file.content_type
    elif text is not None:
        data = text.encode("utf-8")
        name = filename or f"{external_id or 'document'}.{document_type or 'txt'}"
        mime = "text/plain"
    else:
        raise ValidationError("Provide either 'file' or 'text'")
    if len(data) > max_bytes:
        raise ValidationError(f"Document exceeds the maximum size of {max_bytes} bytes")

    meta = _parse_json(metadata, "metadata", dict)
    perms = _parse_json(permissions, "permissions", list)
    if perms is not None and not all(isinstance(p, str) and p for p in perms):
        raise ValidationError("'permissions' must be a list of non-empty strings")

    result = await container.documents.upload(
        session,
        tenant_id=tenant.tenant_id,
        knowledge_base_id=knowledge_base_id,
        data=data,
        filename=name,
        mime_type=mime,
        external_id=external_id,
        document_type=document_type,
        source=source,
        metadata=meta,
        permissions=perms,
        force=force,
    )
    await session.commit()
    if result.job is not None:
        container.worker.notify()
    return DocumentUploadOut(
        outcome=result.outcome,
        document=document_out(result.document, []),
        version=version_out(result.version),
        job=job_out(result.job) if result.job else None,
    )


async def _load_document(session: AsyncSession, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    doc = (
        await session.execute(
            select(Document).options(selectinload(Document.versions)).where(Document.id == document_id)
        )
    ).scalar_one_or_none()
    if doc is None or doc.tenant_id != tenant_id:
        raise NotFoundError(f"Document {document_id} not found")
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    doc = await _load_document(session, tenant.tenant_id, document_id)
    return document_out(doc, doc.versions)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> None:
    doc = await _load_document(session, tenant.tenant_id, document_id)
    kb = await get_kb(session, tenant.tenant_id, doc.knowledge_base_id)
    emb = await session.get(EmbeddingConfig, kb.embedding_config_id)
    assert emb is not None
    await container.vector_store.delete_by_document(collection_of(emb).name, tenant.tenant_id, doc.id)
    await session.delete(doc)
    await session.commit()

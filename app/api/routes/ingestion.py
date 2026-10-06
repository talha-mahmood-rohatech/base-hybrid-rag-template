from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import TenantContext, get_container, get_session, get_tenant
from app.api.serializers import job_out
from app.container import Container
from app.core.errors import NotFoundError, ValidationError
from app.models import Document, DocumentVersion, IngestionJob
from app.schemas.api import IngestionJobCreate, IngestionJobOut

router = APIRouter(prefix="/v1/ingestion/jobs", tags=["ingestion"])


@router.post("", response_model=IngestionJobOut, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    body: IngestionJobCreate,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> IngestionJobOut:
    """(Re-)ingest a document version. With ``document_id`` the latest version is used."""
    if bool(body.document_id) == bool(body.document_version_id):
        raise ValidationError("Provide exactly one of 'document_id' or 'document_version_id'")
    if body.document_version_id:
        version = await session.get(DocumentVersion, body.document_version_id)
    else:
        doc = await session.get(Document, body.document_id)
        if doc is None or doc.tenant_id != tenant.tenant_id:
            raise NotFoundError("Document not found")
        version = (
            await session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == doc.id)
                .order_by(DocumentVersion.version_number.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    if version is None or version.tenant_id != tenant.tenant_id:
        raise NotFoundError("Document version not found")
    job = await container.documents.enqueue(session, version)
    await session.commit()
    container.worker.notify()
    return job_out(job)


@router.get("/{job_id}", response_model=IngestionJobOut)
async def get_job(
    job_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
    session: AsyncSession = Depends(get_session),
) -> IngestionJobOut:
    job = await session.get(IngestionJob, job_id)
    if job is None or job.tenant_id != tenant.tenant_id:
        raise NotFoundError("Ingestion job not found")
    return job_out(job)

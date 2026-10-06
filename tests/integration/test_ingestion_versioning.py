"""Checksums, idempotent ingestion, versioning and active/inactive switching (PostgreSQL + Qdrant)."""

import pytest
from sqlalchemy import func, select

from app.ingestion.embedding_configs import collection_of
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus, VersionStatus
from app.rag.orchestrator import QueryOptions
from app.retrieval.types import RetrievalScope
from scripts.sample_docs import SLA_PAGES, make_pdf
from tests.integration.helpers import ingest, make_workspace

pytestmark = pytest.mark.integration

V1 = "# Holidays\n\nEmployees receive 25 days of paid vacation per year."
V2 = "# Holidays\n\nEmployees receive 28 days of paid vacation per year from 2027."


async def _collection(container):
    async with container.session_factory() as s:
        return collection_of(await container.embedding_configs.get_default(s))


async def test_full_ingestion_writes_postgres_and_qdrant(container):
    ws = await make_workspace(container)
    res = await ingest(
        container,
        ws,
        "sla.pdf",
        make_pdf(SLA_PAGES),
        metadata={"department": "legal"},
        source="contracts",
        permissions=["group:legal"],
    )
    assert res.outcome == "created"
    async with container.session_factory() as s:
        version = await s.get(DocumentVersion, res.version.id)
        job = (
            await s.execute(select(IngestionJob).where(IngestionJob.document_version_id == version.id))
        ).scalar_one()
        chunks = (
            (await s.execute(select(Chunk).where(Chunk.document_version_id == version.id))).scalars().all()
        )
    assert version.status == VersionStatus.ready and version.is_active and version.page_count == 3
    assert job.status == JobStatus.succeeded and job.stats["chunks"] == len(chunks)
    assert job.stats["embedding_dimension"] == 1024
    assert all(c.is_active and c.page in (1, 2, 3) for c in chunks)
    assert all(c.metadata_ == {"department": "legal"} and c.permissions == ["group:legal"] for c in chunks)
    assert len(version.checksum) == 64

    col = await _collection(container)
    scope = RetrievalScope(ws.tenant_id, ws.kb_id)
    assert await container.vector_store.count(col.name, scope) == len(chunks)
    payload = (await container.vector_store.retrieve(col.name, [chunks[0].id]))[0].payload
    for key in (
        "tenant_id",
        "knowledge_base_id",
        "document_id",
        "document_version_id",
        "chunk_id",
        "document_type",
        "source",
        "page",
        "section",
        "permissions",
    ):
        assert key in payload, key
    assert payload["tenant_id"] == str(ws.tenant_id) and payload["document_type"] == "pdf"
    assert payload["source"] == "contracts" and payload["permissions"] == ["group:legal"]


async def test_idempotent_reupload_and_versioning(container):
    ws = await make_workspace(container)
    col = await _collection(container)
    scope = RetrievalScope(ws.tenant_id, ws.kb_id)

    r1 = await ingest(container, ws, "holidays.md", V1)
    again = await ingest(container, ws, "holidays.md", V1)
    assert again.outcome == "unchanged" and again.job is None and again.version.id == r1.version.id

    r2 = await ingest(container, ws, "holidays.md", V2)
    assert r2.outcome == "new_version" and r2.version.version_number == 2
    assert r2.document.id == r1.document.id

    async with container.session_factory() as s:
        v1 = await s.get(DocumentVersion, r1.version.id)
        v2 = await s.get(DocumentVersion, r2.version.id)
        active_chunks = (
            await s.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == r1.document.id, Chunk.is_active)
            )
        ).scalar()
    assert not v1.is_active and v2.is_active
    n_v2 = v2.chunk_count
    assert active_chunks == n_v2
    assert await container.vector_store.count(col.name, scope) == n_v2  # Qdrant flags flipped too

    res = await container.orchestrator.query(
        tenant_id=ws.tenant_id,
        knowledge_base_id=ws.kb_id,
        query="How many vacation days?",
        filters={},
        options=QueryOptions.from_settings(container.settings.retrieval),
    )
    assert "28 days" in res.answer and "25 days" not in res.answer


async def test_forced_reingestion_is_idempotent(container):
    ws = await make_workspace(container)
    col = await _collection(container)
    scope = RetrievalScope(ws.tenant_id, ws.kb_id, active_only=False)
    r1 = await ingest(container, ws, "a.md", V1)
    before = await container.vector_store.count(col.name, scope)
    r2 = await ingest(container, ws, "a.md", V1, force=True)
    assert r2.outcome == "requeued" and r2.version.id == r1.version.id
    assert await container.vector_store.count(col.name, scope) == before  # same deterministic ids, no dupes
    async with container.session_factory() as s:
        n = (
            await s.execute(
                select(func.count()).select_from(Chunk).where(Chunk.document_version_id == r1.version.id)
            )
        ).scalar()
    assert n == before


async def test_failed_ingestion_is_recorded_and_retryable(container):
    ws = await make_workspace(container)
    res = await ingest(container, ws, "broken.pdf", b"%PDF-1.4 garbage")
    async with container.session_factory() as s:
        job = await s.get(IngestionJob, res.job.id)
        job.available_at = func.now()  # skip backoff for the test
        await s.commit()
    for _ in range(5):
        await container.worker.drain()
        async with container.session_factory() as s:
            job = await s.get(IngestionJob, res.job.id)
            if job.status == JobStatus.failed:
                break
            job.available_at = func.now()
            await s.commit()
    async with container.session_factory() as s:
        job = await s.get(IngestionJob, res.job.id)
        version = await s.get(DocumentVersion, res.version.id)
    assert job.status == JobStatus.failed and job.attempts == job.max_attempts
    assert "DocumentParseError" in job.error
    assert version.status == VersionStatus.failed and not version.is_active

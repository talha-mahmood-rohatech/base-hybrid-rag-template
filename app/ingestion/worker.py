"""PostgreSQL-backed ingestion job worker (no Redis/Kafka needed).

Jobs are claimed with ``FOR UPDATE SKIP LOCKED`` so any number of workers (embedded in
the API process or standalone via ``python -m app.ingestion.worker``) can run safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ingestion.pipeline import IngestionPipeline
from app.models import DocumentVersion, IngestionJob, JobStatus, VersionStatus

logger = logging.getLogger(__name__)

_CLAIM_SQL = text(
    """
    UPDATE ingestion_jobs SET status = 'running', attempts = attempts + 1,
           started_at = now(), locked_by = :worker, error = NULL
    WHERE id = (
        SELECT id FROM ingestion_jobs
        WHERE status = 'queued' AND available_at <= now()
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id, document_version_id, attempts, max_attempts
    """
)

_REQUEUE_STALE_SQL = text(
    """
    UPDATE ingestion_jobs SET status = 'queued', locked_by = NULL, available_at = now()
    WHERE status = 'running' AND started_at < now() - make_interval(secs => :lease)
    RETURNING id
    """
)


class IngestionWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        pipeline: IngestionPipeline,
        *,
        poll_interval_s: float = 1.0,
        lease_timeout_s: int = 900,
        worker_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.pipeline = pipeline
        self.poll_interval_s = poll_interval_s
        self.lease_timeout_s = lease_timeout_s
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._wake = asyncio.Event()

    def notify(self) -> None:
        """Wake the worker immediately (e.g. after a job was enqueued in-process)."""
        self._wake.set()

    async def requeue_stale(self) -> int:
        async with self.session_factory() as session, session.begin():
            rows = (await session.execute(_REQUEUE_STALE_SQL, {"lease": self.lease_timeout_s})).all()
        if rows:
            logger.warning("requeued stale ingestion jobs", extra={"count": len(rows)})
        return len(rows)

    async def run_once(self) -> uuid.UUID | None:
        """Claim and process one job. Returns the job id, or None if the queue is empty."""
        async with self.session_factory() as session, session.begin():
            row = (await session.execute(_CLAIM_SQL, {"worker": self.worker_id})).first()
        if row is None:
            return None
        job_id, version_id, attempts, max_attempts = row
        log = {"job_id": str(job_id), "version_id": str(version_id), "attempt": attempts}
        logger.info("ingestion job started", extra=log)
        try:
            result = await self.pipeline.run(version_id)
        except Exception as exc:
            logger.exception("ingestion job failed", extra=log)
            final = attempts >= max_attempts
            async with self.session_factory() as session, session.begin():
                await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == job_id)
                    .values(
                        status=JobStatus.failed if final else JobStatus.queued,
                        error=f"{type(exc).__name__}: {exc}"[:4000],
                        finished_at=datetime.now(UTC) if final else None,
                        available_at=datetime.now(UTC) + timedelta(seconds=min(300, 5 * 2**attempts)),
                        locked_by=None,
                    )
                )
                await session.execute(
                    update(DocumentVersion)
                    .where(DocumentVersion.id == version_id)
                    .values(
                        status=VersionStatus.failed if final else VersionStatus.pending,
                        error=f"{type(exc).__name__}: {exc}"[:4000],
                    )
                )
            return job_id

        async with self.session_factory() as session, session.begin():
            await session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(
                    status=JobStatus.succeeded,
                    finished_at=datetime.now(UTC),
                    locked_by=None,
                    stats={**result.stats, "activated": result.activated, "timings_ms": result.timings_ms},
                )
            )
        logger.info("ingestion job succeeded", extra={**log, "chunks": result.chunk_count})
        return job_id

    async def drain(self, max_jobs: int = 10_000) -> int:
        """Process jobs until the queue is empty (used by tests, CLI and eval)."""
        n = 0
        while n < max_jobs and await self.run_once() is not None:
            n += 1
        return n

    async def run_forever(self, stop: asyncio.Event) -> None:
        logger.info("ingestion worker running", extra={"worker_id": self.worker_id})
        await self.requeue_stale()
        last_stale_check = asyncio.get_running_loop().time()
        while not stop.is_set():
            try:
                processed = await self.run_once()
                now = asyncio.get_running_loop().time()
                if now - last_stale_check > 60:
                    await self.requeue_stale()
                    last_stale_check = now
            except Exception:
                logger.exception("worker loop error")
                processed = None
            if processed is None:
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self.poll_interval_s)


async def _main() -> None:  # pragma: no cover - process entrypoint
    import signal

    from app.container import Container
    from app.core.config import get_settings
    from app.core.logging import configure_logging

    settings = get_settings()
    configure_logging(settings.log_level)
    container = await Container.create(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, stop.set)
    try:
        await container.worker.run_forever(stop)
    finally:
        await container.aclose()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())

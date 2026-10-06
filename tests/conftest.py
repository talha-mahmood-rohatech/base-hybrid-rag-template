from __future__ import annotations

import asyncio
import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_CONFIG_FILE", str(ROOT / "config" / "rag.yaml"))

from app.core.config import (  # noqa: E402
    ChunkingSettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    QdrantSettings,
    RerankerSettings,
    Settings,
    StorageSettings,
    WorkerSettings,
)

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:15432/rag_test"
)
TEST_QDRANT_URL = os.environ.get("TEST_QDRANT_URL", "http://localhost:6333")


def _reachable(url: str, default_port: int) -> bool:
    u = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=1.5):
            return True
    except OSError:
        return False


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "environment": "test",
        "database": DatabaseSettings(url=TEST_DATABASE_URL, pool_size=5),
        "qdrant": QdrantSettings(url=TEST_QDRANT_URL, collection_prefix=f"test_{uuid.uuid4().hex[:8]}"),
        "embedding": EmbeddingSettings(provider="hashing", model="hashing-v1", dimension=1024),
        "chunking": ChunkingSettings(chunk_size=120, chunk_overlap=20),
        "reranker": RerankerSettings(provider="none"),
        "llm": LLMSettings(provider="extractive", model="extractive-v1"),
        "storage": StorageSettings(blob_dir=str(tmp_path / "blobs")),
        "worker": WorkerSettings(embedded=False),
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture(scope="session")
def services_available() -> bool:
    return _reachable(TEST_DATABASE_URL, 5432) and _reachable(TEST_QDRANT_URL, 6333)


@pytest.fixture(scope="session")
def migrated_db(services_available):
    """Run migrations from scratch on the test database (also tests downgrade/upgrade)."""
    if not services_available:
        pytest.skip("PostgreSQL/Qdrant not reachable; run `docker compose up -d postgres qdrant`")
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.attributes["database_url"] = TEST_DATABASE_URL

    async def _run() -> None:
        # alembic env.py uses asyncio.run internally -> execute in a worker thread
        await asyncio.to_thread(command.downgrade, cfg, "base")
        await asyncio.to_thread(command.upgrade, cfg, "head")

    asyncio.run(_run())
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("rag")


@pytest.fixture(scope="session")
async def container(migrated_db, session_tmp):
    """Fully wired platform against real PostgreSQL + Qdrant, deterministic model doubles."""
    from app.container import Container
    from tests.fakes import LexicalReranker

    settings = make_settings(session_tmp)
    c = await Container.create(settings, reranker=LexicalReranker())
    yield c
    # Clean up this session's Qdrant collections.
    client = c.vector_store.client  # type: ignore[attr-defined]
    for col in (await client.get_collections()).collections:
        if col.name.startswith(settings.qdrant.collection_prefix):
            await client.delete_collection(col.name)
    await c.aclose()

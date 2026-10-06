"""End-to-end with the real local models: mxbai-embed-large (1024d) + jina cross-encoder reranker.

Opt-in (slow, needs the models in the fastembed cache):
    RUN_MODEL_TESTS=1 FASTEMBED_CACHE_PATH=.cache/models pytest -m models
"""

import os

import pytest

from app.core.config import ChunkingSettings, EmbeddingSettings, RerankerSettings
from app.rag.orchestrator import QueryOptions
from scripts.sample_docs import sample_files
from tests.conftest import make_settings
from tests.integration.helpers import ingest, make_workspace

pytestmark = [
    pytest.mark.integration,
    pytest.mark.models,
    pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run"),
]


@pytest.fixture(scope="module")
async def real_container(migrated_db, tmp_path_factory):
    from app.container import Container

    settings = make_settings(
        tmp_path_factory.mktemp("real"),
        embedding=EmbeddingSettings(
            provider="fastembed", model="mixedbread-ai/mxbai-embed-large-v1", dimension=1024
        ),
        reranker=RerankerSettings(provider="fastembed", model="jinaai/jina-reranker-v2-base-multilingual"),
        chunking=ChunkingSettings(chunk_size=480, chunk_overlap=72),
    )
    c = await Container.create(settings)
    yield c
    client = c.vector_store.client  # type: ignore[attr-defined]
    for col in (await client.get_collections()).collections:
        if col.name.startswith(settings.qdrant.collection_prefix):
            await client.delete_collection(col.name)
    await c.aclose()


async def test_real_models_pipeline(real_container):
    c = real_container
    ws = await make_workspace(c)
    for name, data in sample_files().items():
        await ingest(c, ws, name, data)

    # A paraphrased question with little lexical overlap: semantic retrieval + cross-encoder must carry it.
    res = await c.orchestrator.query(
        tenant_id=ws.tenant_id,
        knowledge_base_id=ws.kb_id,
        query="If I quit my yearly plan early, do I get anything back?",
        filters={},
        options=QueryOptions.from_settings(c.settings.retrieval),
    )
    top = res.results[0]
    assert top.chunk.document_name == "refund_policy.md"
    assert "prorated" in top.chunk.text
    assert top.reranker_score is not None
    scores = [x.reranker_score for x in res.results]
    assert scores == sorted(scores, reverse=True)
    assert res.counts["dense_candidates"] > 0 and res.counts["sparse_candidates"] > 0
    assert res.citations and all(ci.document_name for ci in res.citations)

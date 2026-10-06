"""Embedding dimension handling + QdrantVectorStore behaviour on Qdrant's in-process engine."""

import math
import uuid
from collections.abc import Sequence

import pytest

from app.core.errors import EmbeddingConfigMismatchError, VectorDimensionMismatchError
from app.providers.embeddings.base import EmbeddingProvider, EmbeddingSpec
from app.providers.embeddings.hashing_provider import HashingEmbeddingProvider
from app.providers.vectorstores.base import CollectionSpec, VectorPoint
from app.providers.vectorstores.qdrant import QdrantVectorStore, collection_name_for
from app.retrieval.filters import parse_filters
from app.retrieval.types import RetrievalScope


def spec(dim: int) -> EmbeddingSpec:
    return EmbeddingSpec("hashing", "hashing-v1", dim, "v1", "cosine")


@pytest.mark.parametrize("dim", [1024, 1536, 3072])
async def test_configurable_high_dimensions(dim):
    p = HashingEmbeddingProvider(spec(dim))
    vecs = await p.embed_documents(["refund policy", "security incident"])
    q = await p.embed_query("refund policy")
    assert all(len(v) == dim for v in vecs) and len(q) == dim
    assert math.isclose(sum(x * x for x in q), 1.0, rel_tol=1e-9)
    assert vecs[0] == q  # deterministic


async def test_provider_output_dimension_is_validated():
    class Broken(EmbeddingProvider):
        async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
            return [[0.1] * 768 for _ in texts]

        async def _embed_query(self, text: str) -> Sequence[float]:
            return [0.1] * 768

    with pytest.raises(VectorDimensionMismatchError):
        await Broken(spec(1024)).embed_documents(["x"])
    with pytest.raises(VectorDimensionMismatchError):
        await Broken(spec(1024)).embed_query("x")


def test_collection_naming_is_per_embedding_config():
    a = collection_name_for("rag", "openai", "text-embedding-3-large", 3072, "cosine", "v1")
    b = collection_name_for("rag", "openai", "text-embedding-3-large", 1536, "cosine", "v1")
    c = collection_name_for("rag", "openai", "text-embedding-3-large", 3072, "cosine", "v2")
    assert len({a, b, c}) == 3
    assert a == "rag__openai__text_embedding_3_large__3072d__cosine__v1"


# --- Qdrant (in-process engine; the same code path is exercised against a real server in integration tests)
T1, T2, KB1, KB2 = (uuid.uuid4() for _ in range(4))


def point(tenant, kb, text, *, dim=8, active=True, **payload):
    vec = [0.0] * dim
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    pid = uuid.uuid4()
    doc = payload.pop("document_id", str(uuid.uuid4()))
    return VectorPoint(
        id=pid,
        vector=vec,
        payload={
            "tenant_id": str(tenant),
            "knowledge_base_id": str(kb),
            "document_id": doc,
            "document_version_id": payload.pop("document_version_id", str(uuid.uuid4())),
            "chunk_id": str(pid),
            "is_active": active,
            "permissions": ["public"],
            **payload,
        },
    )


@pytest.fixture
async def store():
    s = QdrantVectorStore.from_url(":memory:")
    yield s
    await s.aclose()


SPEC = CollectionSpec("unit_chunks", 8, "cosine")


async def test_upsert_rejects_wrong_dimension(store):
    await store.ensure_collection(SPEC)
    with pytest.raises(VectorDimensionMismatchError):
        await store.upsert(SPEC, [point(T1, KB1, "abc", dim=9)])


async def test_upsert_requires_isolation_payload(store):
    p = point(T1, KB1, "abc")
    del p.payload["tenant_id"]
    with pytest.raises(ValueError):
        await store.upsert(SPEC, [p])


async def test_existing_collection_with_other_dimension_is_rejected(store):
    await store.ensure_collection(SPEC)
    store._known.clear()
    with pytest.raises(EmbeddingConfigMismatchError):
        await store.ensure_collection(CollectionSpec("unit_chunks", 16, "cosine"))


async def test_search_is_scoped_to_tenant_and_kb(store):
    pts = [
        point(T1, KB1, "refund policy", document_type="md"),
        point(T1, KB2, "refund policy", document_type="md"),
        point(T2, KB1, "refund policy", document_type="md"),
        point(T1, KB1, "refund policy old", active=False),
    ]
    await store.upsert(SPEC, pts)
    hits = await store.search(SPEC.name, pts[0].vector, RetrievalScope(T1, KB1), top_k=10)
    assert [h.id for h in hits] == [pts[0].id]
    hits_t2 = await store.search(SPEC.name, pts[0].vector, RetrievalScope(T2, KB1), top_k=10)
    assert [h.id for h in hits_t2] == [pts[2].id]
    assert await store.count(SPEC.name, RetrievalScope(T1, KB1)) == 1


async def test_metadata_filters(store):
    pts = [
        point(T1, KB1, "a", document_type="pdf", page=1, metadata={"dept": "finance"}),
        point(T1, KB1, "a", document_type="pdf", page=7, metadata={"dept": "legal"}),
        point(
            T1,
            KB1,
            "a",
            document_type="md",
            page=None,
            metadata={"dept": "finance"},
            permissions=["group:hr"],
        ),
    ]
    await store.upsert(SPEC, pts)
    scope = RetrievalScope(T1, KB1)

    async def ids(f):
        return {h.id for h in await store.search(SPEC.name, pts[0].vector, scope, parse_filters(f), 10)}

    assert await ids({"document_type": "pdf"}) == {pts[0].id, pts[1].id}
    assert await ids({"page": {"gte": 5}}) == {pts[1].id}
    assert await ids({"metadata.dept": "finance"}) == {pts[0].id, pts[2].id}
    assert await ids({"permissions": ["group:hr"]}) == {pts[2].id}
    assert await ids({"document_id": pts[1].payload["document_id"]}) == {pts[1].id}


async def test_set_active_and_delete_by_version(store):
    version = str(uuid.uuid4())
    pts = [point(T1, KB1, "x", active=False, document_version_id=version) for _ in range(3)]
    other_tenant = point(T2, KB1, "x", active=False, document_version_id=version)
    await store.upsert(SPEC, [*pts, other_tenant])
    scope = RetrievalScope(T1, KB1)
    assert await store.count(SPEC.name, scope) == 0
    await store.set_active(SPEC.name, T1, uuid.UUID(version), True)
    assert await store.count(SPEC.name, scope) == 3
    # tenant-scoped: the other tenant's point with the same version id is untouched
    assert await store.count(SPEC.name, RetrievalScope(T2, KB1)) == 0
    await store.delete_by_document_version(SPEC.name, T1, uuid.UUID(version))
    assert await store.count(SPEC.name, RetrievalScope(T1, KB1, active_only=False)) == 0
    assert await store.count(SPEC.name, RetrievalScope(T2, KB1, active_only=False)) == 1

"""Qdrant implementation of :class:`VectorStore`.

Collection strategy: one collection per *embedding configuration*
(provider/model/dimension/distance/version), shared by all tenants and knowledge bases.
Isolation is enforced with mandatory payload filters on ``tenant_id`` (indexed with
``is_tenant=True`` so Qdrant co-locates each tenant's vectors) and ``knowledge_base_id``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.errors import EmbeddingConfigMismatchError, ProviderError, VectorDimensionMismatchError
from app.core.utils import slugify
from app.providers.vectorstores.base import CollectionSpec, VectorHit, VectorPoint, VectorStore
from app.retrieval.filters import FilterCondition
from app.retrieval.types import RetrievalScope

logger = logging.getLogger(__name__)

DISTANCES = {"cosine": qm.Distance.COSINE, "dot": qm.Distance.DOT, "euclid": qm.Distance.EUCLID}

PAYLOAD_INDEXES: dict[str, Any] = {
    "tenant_id": qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True),
    "knowledge_base_id": qm.PayloadSchemaType.KEYWORD,
    "document_id": qm.PayloadSchemaType.KEYWORD,
    "document_version_id": qm.PayloadSchemaType.KEYWORD,
    "chunk_id": qm.PayloadSchemaType.KEYWORD,
    "document_type": qm.PayloadSchemaType.KEYWORD,
    "source": qm.PayloadSchemaType.KEYWORD,
    "section": qm.PayloadSchemaType.KEYWORD,
    "permissions": qm.PayloadSchemaType.KEYWORD,
    "page": qm.PayloadSchemaType.INTEGER,
    "is_active": qm.PayloadSchemaType.BOOL,
}


def collection_name_for(
    prefix: str, provider: str, model: str, dimension: int, distance: str, version: str
) -> str:
    name = f"{prefix}__{slugify(provider, 20)}__{slugify(model, 80)}__{dimension}d__{distance}__{slugify(version, 20)}"
    return name.replace("-", "_")[:255]


def _match(key: str, value: Any) -> qm.FieldCondition:
    if isinstance(value, float) and not value.is_integer():
        return qm.FieldCondition(key=key, range=qm.Range(gte=value, lte=value))
    if isinstance(value, float):
        value = int(value)
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


def build_qdrant_filter(scope: RetrievalScope, filters: Sequence[FilterCondition] = ()) -> qm.Filter:
    must: list[Any] = [
        qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(scope.tenant_id))),
        qm.FieldCondition(key="knowledge_base_id", match=qm.MatchValue(value=str(scope.knowledge_base_id))),
    ]
    if scope.active_only:
        must.append(qm.FieldCondition(key="is_active", match=qm.MatchValue(value=True)))
    for cond in filters:
        key = cond.field  # "metadata.foo" is a valid nested payload path in Qdrant
        if cond.op == "eq":
            must.append(_match(key, cond.value))
        elif cond.op in ("in", "any"):
            values = list(cond.value)
            if all(isinstance(v, str) for v in values) or all(
                isinstance(v, int) and not isinstance(v, bool) for v in values
            ):
                must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=values)))
            else:
                must.append(qm.Filter(should=[_match(key, v) for v in values]))
        elif cond.op == "range":
            must.append(qm.FieldCondition(key=key, range=qm.Range(**cond.value)))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported filter op {cond.op}")
    return qm.Filter(must=must)


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        search_hnsw_ef: int | None = 128,
        on_disk_vectors: bool = False,
        upsert_batch_size: int = 128,
    ) -> None:
        self.client = client
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.search_hnsw_ef = search_hnsw_ef
        self.on_disk_vectors = on_disk_vectors
        self.upsert_batch_size = upsert_batch_size
        self._known: dict[str, CollectionSpec] = {}

    @classmethod
    def from_url(cls, url: str, api_key: str | None = None, timeout_s: float = 30.0, **kw: Any):
        if url == ":memory:":
            return cls(AsyncQdrantClient(location=":memory:"), **kw)
        return cls(AsyncQdrantClient(url=url, api_key=api_key, timeout=int(timeout_s)), **kw)

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        if self._known.get(spec.name) == spec:
            return
        distance = DISTANCES[spec.distance_metric]
        try:
            if await self.client.collection_exists(spec.name):
                info = await self.client.get_collection(spec.name)
                vectors = info.config.params.vectors
                if vectors is None or isinstance(vectors, dict):  # named vectors - not created by us
                    raise EmbeddingConfigMismatchError(f"Collection {spec.name} uses named vectors")
                if vectors.size != spec.dimension or vectors.distance != distance:
                    raise EmbeddingConfigMismatchError(
                        f"Collection {spec.name} has size={vectors.size} distance={vectors.distance}, "
                        f"expected size={spec.dimension} distance={distance}",
                    )
            else:
                logger.info(
                    "creating qdrant collection", extra={"collection": spec.name, "dim": spec.dimension}
                )
                try:
                    await self.client.create_collection(
                        collection_name=spec.name,
                        vectors_config=qm.VectorParams(
                            size=spec.dimension, distance=distance, on_disk=self.on_disk_vectors
                        ),
                        hnsw_config=qm.HnswConfigDiff(
                            m=self.hnsw_m, ef_construct=self.hnsw_ef_construct, payload_m=self.hnsw_m
                        ),
                    )
                except (UnexpectedResponse, ValueError) as exc:
                    # Concurrent creator won the race.
                    if not await self.client.collection_exists(spec.name):
                        raise
                    logger.debug("collection created concurrently: %s", exc)
                for field_name, schema in PAYLOAD_INDEXES.items():
                    await self.client.create_payload_index(
                        spec.name, field_name=field_name, field_schema=schema, wait=True
                    )
        except EmbeddingConfigMismatchError:
            raise
        except Exception as exc:
            raise ProviderError(f"Qdrant ensure_collection failed: {exc}") from exc
        self._known[spec.name] = spec

    async def upsert(self, spec: CollectionSpec, points: Sequence[VectorPoint]) -> None:
        for p in points:
            if len(p.vector) != spec.dimension:
                raise VectorDimensionMismatchError(
                    f"Point {p.id} has dimension {len(p.vector)}; collection {spec.name} expects {spec.dimension}",
                    details={"expected": spec.dimension, "actual": len(p.vector)},
                )
            for required in ("tenant_id", "knowledge_base_id", "document_id", "chunk_id"):
                if not p.payload.get(required):
                    raise ValueError(f"Point {p.id} payload is missing required '{required}'")
        await self.ensure_collection(spec)
        try:
            for i in range(0, len(points), self.upsert_batch_size):
                batch = points[i : i + self.upsert_batch_size]
                await self.client.upsert(
                    collection_name=spec.name,
                    points=[qm.PointStruct(id=str(p.id), vector=p.vector, payload=p.payload) for p in batch],
                    wait=True,
                )
        except Exception as exc:
            raise ProviderError(f"Qdrant upsert failed: {exc}") from exc

    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        scope: RetrievalScope,
        filters: Sequence[FilterCondition] = (),
        top_k: int = 50,
    ) -> list[VectorHit]:
        spec = self._known.get(collection)
        if spec is not None and len(vector) != spec.dimension:
            raise VectorDimensionMismatchError(
                f"Query vector dimension {len(vector)} != collection dimension {spec.dimension}"
            )
        try:
            if not await self.client.collection_exists(collection):
                return []
            res = await self.client.query_points(
                collection_name=collection,
                query=list(vector),
                query_filter=build_qdrant_filter(scope, filters),
                limit=top_k,
                with_payload=True,
                search_params=qm.SearchParams(hnsw_ef=self.search_hnsw_ef) if self.search_hnsw_ef else None,
            )
        except Exception as exc:
            raise ProviderError(f"Qdrant search failed: {exc}") from exc
        hits = []
        for p in res.points:
            payload = p.payload or {}
            # Defence in depth: never return a point outside the requested scope.
            if payload.get("tenant_id") != str(scope.tenant_id) or payload.get("knowledge_base_id") != str(
                scope.knowledge_base_id
            ):
                logger.error("qdrant returned out-of-scope point; dropping", extra={"point": str(p.id)})
                continue
            hits.append(VectorHit(id=uuid.UUID(str(p.id)), score=float(p.score), payload=payload))
        return hits

    def _version_filter(self, tenant_id: uuid.UUID, key: str, value: uuid.UUID) -> qm.Filter:
        return qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id))),
                qm.FieldCondition(key=key, match=qm.MatchValue(value=str(value))),
            ]
        )

    async def set_active(
        self, collection: str, tenant_id: uuid.UUID, document_version_id: uuid.UUID, active: bool
    ) -> None:
        if not await self.client.collection_exists(collection):
            return
        await self.client.set_payload(
            collection_name=collection,
            payload={"is_active": active},
            points=qm.FilterSelector(
                filter=self._version_filter(tenant_id, "document_version_id", document_version_id)
            ),
            wait=True,
        )

    async def delete_by_document_version(
        self, collection: str, tenant_id: uuid.UUID, document_version_id: uuid.UUID
    ) -> None:
        if not await self.client.collection_exists(collection):
            return
        await self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=self._version_filter(tenant_id, "document_version_id", document_version_id)
            ),
            wait=True,
        )

    async def delete_by_document(self, collection: str, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
        if not await self.client.collection_exists(collection):
            return
        await self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=self._version_filter(tenant_id, "document_id", document_id)
            ),
            wait=True,
        )

    async def delete_by_tenant(self, collection: str, tenant_id: uuid.UUID) -> None:
        if not await self.client.collection_exists(collection):
            return
        await self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id)))]
                )
            ),
            wait=True,
        )

    async def count(
        self, collection: str, scope: RetrievalScope, filters: Sequence[FilterCondition] = ()
    ) -> int:
        if not await self.client.collection_exists(collection):
            return 0
        res = await self.client.count(
            collection, count_filter=build_qdrant_filter(scope, filters), exact=True
        )
        return res.count

    async def retrieve(self, collection: str, ids: Sequence[uuid.UUID]) -> list[VectorHit]:
        if not ids or not await self.client.collection_exists(collection):
            return []
        recs = await self.client.retrieve(collection, ids=[str(i) for i in ids], with_payload=True)
        return [VectorHit(id=uuid.UUID(str(r.id)), score=0.0, payload=r.payload or {}) for r in recs]

    async def healthcheck(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        await self.client.close()

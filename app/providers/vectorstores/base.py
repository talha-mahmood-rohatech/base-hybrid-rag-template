from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.filters import FilterCondition
from app.retrieval.types import RetrievalScope


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    name: str
    dimension: int
    distance_metric: str  # cosine | dot | euclid


@dataclass(slots=True)
class VectorPoint:
    id: uuid.UUID
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorHit:
    id: uuid.UUID
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    """Dense vector index. Every search is bounded by a :class:`RetrievalScope`."""

    @abstractmethod
    async def ensure_collection(self, spec: CollectionSpec) -> None: ...

    @abstractmethod
    async def upsert(self, spec: CollectionSpec, points: Sequence[VectorPoint]) -> None: ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: Sequence[float],
        scope: RetrievalScope,
        filters: Sequence[FilterCondition] = (),
        top_k: int = 50,
    ) -> list[VectorHit]: ...

    @abstractmethod
    async def set_active(
        self, collection: str, tenant_id: uuid.UUID, document_version_id: uuid.UUID, active: bool
    ) -> None: ...

    @abstractmethod
    async def delete_by_document_version(
        self, collection: str, tenant_id: uuid.UUID, document_version_id: uuid.UUID
    ) -> None: ...

    @abstractmethod
    async def delete_by_document(
        self, collection: str, tenant_id: uuid.UUID, document_id: uuid.UUID
    ) -> None: ...

    @abstractmethod
    async def count(
        self, collection: str, scope: RetrievalScope, filters: Sequence[FilterCondition] = ()
    ) -> int: ...

    @abstractmethod
    async def retrieve(self, collection: str, ids: Sequence[uuid.UUID]) -> list[VectorHit]: ...

    @abstractmethod
    async def healthcheck(self) -> bool: ...

    async def aclose(self) -> None:  # noqa: B027
        pass

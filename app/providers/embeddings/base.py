from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.errors import VectorDimensionMismatchError


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Identity of an embedding space. Two specs with equal fields are interchangeable."""

    provider: str
    model: str
    dimension: int
    version: str
    distance_metric: str


class EmbeddingProvider(ABC):
    """Turns text into fixed-size dense vectors.

    Implementations must return vectors of exactly ``spec.dimension`` floats; the base
    class enforces this so a misconfigured model can never write into the wrong collection.
    """

    def __init__(self, spec: EmbeddingSpec, *, max_input_tokens: int | None = None) -> None:
        self.spec = spec
        self.max_input_tokens = max_input_tokens

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._embed_documents(list(texts))
        if len(vectors) != len(texts):
            raise VectorDimensionMismatchError(
                f"Provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return [self.validate(v) for v in vectors]

    async def embed_query(self, text: str) -> list[float]:
        return self.validate(await self._embed_query(text))

    def validate(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.spec.dimension:
            raise VectorDimensionMismatchError(
                f"Embedding model '{self.spec.model}' produced dimension {len(vector)}, "
                f"configured dimension is {self.spec.dimension}",
                details={"expected": self.spec.dimension, "actual": len(vector)},
            )
        out = [float(x) for x in vector]
        if not all(math.isfinite(x) for x in out):
            raise VectorDimensionMismatchError("Embedding contains non-finite values")
        return out

    @abstractmethod
    async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]: ...

    @abstractmethod
    async def _embed_query(self, text: str) -> Sequence[float]: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass

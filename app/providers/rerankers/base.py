from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Reranker(ABC):
    """Scores (query, passage) pairs jointly, e.g. with a cross-encoder.

    Returns one relevance score per input document, aligned with the input order.
    Scores are only comparable within a single call.
    """

    provider: str = "base"
    model: str = ""

    @property
    def enabled(self) -> bool:
        return True

    @abstractmethod
    async def score(self, query: str, documents: Sequence[str]) -> list[float]: ...

    async def aclose(self) -> None:  # noqa: B027
        pass


class PassthroughReranker(Reranker):
    """Disables reranking: the fusion order is kept and no reranker score is produced."""

    provider = "none"
    model = "none"

    @property
    def enabled(self) -> bool:
        return False

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise NotImplementedError("PassthroughReranker does not score")

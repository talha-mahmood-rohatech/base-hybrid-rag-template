from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.types import RetrievalQuery, RetrievalResult


class SparseSearch(ABC):
    """Lexical retrieval (BM25 / full-text). Must honour ``query.scope`` and ``query.filters``."""

    name: str = "sparse"

    @abstractmethod
    async def search(self, query: RetrievalQuery, *, language: str = "english") -> list[RetrievalResult]: ...

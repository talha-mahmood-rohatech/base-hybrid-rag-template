from __future__ import annotations

from app.providers.sparse.base import SparseSearch
from app.retrieval.types import RetrievalQuery, RetrievalResult


class SparseRetriever:
    """Lexical retrieval path (BM25 / FTS) - independent of embeddings and Qdrant."""

    name = "sparse"

    def __init__(self, search: SparseSearch, language: str = "english") -> None:
        self.search = search
        self.language = language

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        results = await self.search.search(query, language=self.language)
        for r in results:
            r.retriever = self.name
        return results

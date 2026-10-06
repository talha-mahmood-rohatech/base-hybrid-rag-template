from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from app.providers.embeddings.base import EmbeddingProvider
from app.providers.vectorstores.base import VectorStore
from app.retrieval.types import RetrievalQuery, RetrievalResult


@dataclass(slots=True)
class DenseRetrieval:
    results: list[RetrievalResult]
    embed_ms: float
    search_ms: float


class DenseRetriever:
    """Query embedding -> Qdrant ANN search within the tenant/KB scope."""

    name = "dense"

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore, collection: str) -> None:
        self.embedder = embedder
        self.store = store
        self.collection = collection

    async def retrieve(self, query: RetrievalQuery) -> DenseRetrieval:
        t0 = time.perf_counter()
        vector = await self.embedder.embed_query(query.text)
        t1 = time.perf_counter()
        hits = await self.store.search(
            self.collection, vector, query.scope, filters=query.filters, top_k=query.top_k
        )
        t2 = time.perf_counter()
        results = [
            RetrievalResult(
                chunk_id=h.id,
                document_id=uuid.UUID(h.payload["document_id"]),
                score=h.score,
                rank=i + 1,
                retriever=self.name,
                payload=h.payload,
            )
            for i, h in enumerate(hits)
        ]
        return DenseRetrieval(results, round((t1 - t0) * 1000, 2), round((t2 - t1) * 1000, 2))

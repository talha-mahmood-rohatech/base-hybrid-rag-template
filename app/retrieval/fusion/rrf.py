"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

    RRF(d) = sum_i  w_i / (k + rank_i(d))

Rank-based: raw retriever scores (cosine similarity, BM25) are never compared, which is
exactly why it is robust for combining dense and sparse retrieval. ``k`` (default 60)
dampens the influence of top ranks. ``rank_i`` is 1-based; a document missing from a list
contributes nothing for that list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.retrieval.fusion.base import FusionStrategy, collect, finalize
from app.retrieval.types import Candidate, RetrievalResult


class ReciprocalRankFusion(FusionStrategy):
    name = "rrf"

    def __init__(self, k: int = 60, weights: Mapping[str, float] | None = None) -> None:
        if k < 1:
            raise ValueError("rrf k must be >= 1")
        self.k = k
        self.weights = dict(weights or {})

    def fuse(self, result_lists: Mapping[str, Sequence[RetrievalResult]], top_k: int) -> list[Candidate]:
        by_id = collect(result_lists)
        for cand in by_id.values():
            cand.fused_score = sum(
                self.weights.get(retriever, 1.0) / (self.k + rank) for retriever, rank in cand.ranks.items()
            )
        return finalize(list(by_id.values()), top_k)

    def params(self) -> dict:
        return {"k": self.k, "weights": self.weights or None}

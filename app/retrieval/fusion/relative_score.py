"""Relative score fusion: min-max normalize each retriever's scores, then weighted sum.

Provided as an alternative to RRF for experiments/evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.retrieval.fusion.base import FusionStrategy, collect, finalize
from app.retrieval.types import Candidate, RetrievalResult


class RelativeScoreFusion(FusionStrategy):
    name = "relative_score"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def fuse(self, result_lists: Mapping[str, Sequence[RetrievalResult]], top_k: int) -> list[Candidate]:
        by_id = collect(result_lists)
        norms: dict[str, tuple[float, float]] = {}
        for retriever, results in result_lists.items():
            if results:
                scores = [r.score for r in results]
                norms[retriever] = (min(scores), max(scores))
        for cand in by_id.values():
            total = 0.0
            for retriever, score in cand.scores.items():
                lo, hi = norms[retriever]
                norm = 1.0 if hi == lo else (score - lo) / (hi - lo)
                total += self.weights.get(retriever, 1.0) * norm
            cand.fused_score = total
        return finalize(list(by_id.values()), top_k)

    def params(self) -> dict:
        return {"weights": self.weights or None}

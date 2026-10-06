from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from app.retrieval.types import Candidate, RetrievalResult


class FusionStrategy(ABC):
    """Merges ranked lists from independent retrievers into one ranked candidate list."""

    name: str = "base"

    @abstractmethod
    def fuse(self, result_lists: Mapping[str, Sequence[RetrievalResult]], top_k: int) -> list[Candidate]: ...

    def params(self) -> dict:
        return {}


def collect(result_lists: Mapping[str, Sequence[RetrievalResult]]) -> dict:
    """Index results by chunk id, keeping each retriever's rank and raw score."""
    by_id: dict = {}
    for retriever, results in result_lists.items():
        seen = set()
        for r in results:
            if r.chunk_id in seen:  # a retriever must not vote twice for one chunk
                continue
            seen.add(r.chunk_id)
            cand = by_id.get(r.chunk_id)
            if cand is None:
                cand = by_id[r.chunk_id] = Candidate(
                    chunk_id=r.chunk_id, document_id=r.document_id, fused_score=0.0
                )
            cand.ranks[retriever] = r.rank
            cand.scores[retriever] = r.score
    return by_id


def finalize(cands: list[Candidate], top_k: int) -> list[Candidate]:
    # Deterministic tie-breaking: higher score, then best individual rank, then id.
    cands.sort(key=lambda c: (-c.fused_score, min(c.ranks.values(), default=10**9), str(c.chunk_id)))
    out = cands[:top_k]
    for i, c in enumerate(out, start=1):
        c.fused_rank = i
    return out

"""Reranking stage: cross-encoder scores the fused candidates, keeps the best ``top_n``."""

from __future__ import annotations

from collections.abc import Sequence

from app.providers.rerankers.base import Reranker
from app.retrieval.types import Candidate


def rerank_text(cand: Candidate, include_section: bool) -> str:
    assert cand.chunk is not None
    if include_section and cand.chunk.section:
        return f"{cand.chunk.section}\n{cand.chunk.text}"
    return cand.chunk.text


class RerankingStage:
    def __init__(self, reranker: Reranker, *, include_section: bool = True) -> None:
        self.reranker = reranker
        self.include_section = include_section

    @property
    def enabled(self) -> bool:
        return self.reranker.enabled

    async def run(
        self, query: str, candidates: Sequence[Candidate], top_n: int, *, enabled: bool = True
    ) -> list[Candidate]:
        cands = [c for c in candidates if c.chunk is not None]
        if not cands:
            return []
        if not (enabled and self.reranker.enabled):
            out = sorted(cands, key=lambda c: c.fused_rank)[:top_n]
        else:
            scores = await self.reranker.score(query, [rerank_text(c, self.include_section) for c in cands])
            if len(scores) != len(cands):
                raise ValueError(f"Reranker returned {len(scores)} scores for {len(cands)} candidates")
            for c, s in zip(cands, scores, strict=True):
                c.reranker_score = float(s)
            # Ties broken by fusion rank so behaviour is deterministic.
            out = sorted(cands, key=lambda c: (-(c.reranker_score or 0.0), c.fused_rank))[:top_n]
        for i, c in enumerate(out, start=1):
            c.final_rank = i
        return out

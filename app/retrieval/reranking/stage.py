"""Reranking stage: cross-encoder scores the fused candidates, keeps the best ``top_n``."""

from __future__ import annotations

from collections.abc import Sequence

from app.providers.rerankers.base import Reranker
from app.retrieval.types import Candidate


def rerank_text(cand: Candidate, include_section: bool, max_chars: int | None = None) -> str:
    assert cand.chunk is not None
    text = cand.chunk.text[:max_chars] if max_chars else cand.chunk.text
    if include_section and cand.chunk.section:
        return f"{cand.chunk.section}\n{text}"
    return text


class RerankingStage:
    def __init__(
        self,
        reranker: Reranker,
        *,
        include_section: bool = True,
        max_candidates: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.reranker = reranker
        self.include_section = include_section
        self.max_candidates = max_candidates
        self.max_chars = max_chars

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
            # Only the best-fused candidates are worth the cross-encoder's cost.
            cands = sorted(cands, key=lambda c: c.fused_rank)[: self.max_candidates or len(cands)]
            texts = [rerank_text(c, self.include_section, self.max_chars) for c in cands]
            scores = await self.reranker.score(query, texts)
            if len(scores) != len(cands):
                raise ValueError(f"Reranker returned {len(scores)} scores for {len(cands)} candidates")
            for c, s in zip(cands, scores, strict=True):
                c.reranker_score = float(s)
            # Ties broken by fusion rank so behaviour is deterministic.
            out = sorted(cands, key=lambda c: (-(c.reranker_score or 0.0), c.fused_rank))[:top_n]
        for i, c in enumerate(out, start=1):
            c.final_rank = i
        return out

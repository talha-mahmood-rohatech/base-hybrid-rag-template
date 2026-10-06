"""Deterministic test doubles for model providers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

from app.providers.rerankers.base import Reranker

_W = re.compile(r"[a-z0-9]+")


class LexicalReranker(Reranker):
    """Scores by query-term coverage. Deterministic stand-in for a cross-encoder in tests."""

    provider = "test"
    model = "lexical-overlap"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        q = set(_W.findall(query.lower()))
        out = []
        for d in documents:
            words = _W.findall(d.lower())
            hits = sum(1 for w in q if w in words)
            out.append(hits / (len(q) or 1) + 1.0 / (1 + len(words)))
        return out


class FixedReranker(Reranker):
    provider = "test"
    model = "fixed"

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return self.scores[: len(documents)]


class SlowRetriever:
    """Used to prove dense and sparse run concurrently."""

    def __init__(self, delay: float, results, name: str) -> None:
        self.delay = delay
        self.results = results
        self.name = name

    async def retrieve(self, query):
        await asyncio.sleep(self.delay)
        return self.results

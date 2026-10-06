"""Retrieval and citation metrics (pure functions, no I/O).

``retrieved`` is an ordered list of item ids (best first); ``relevant`` maps id -> graded
relevance (>0 means relevant). Several retrieved chunks may be relevant for one query.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = {i for i, g in relevant.items() if g > 0}
    if not rel:
        return 0.0
    return len(rel & set(retrieved[:k])) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = {i for i, g in relevant.items() if g > 0}
    return sum(1 for i in retrieved[:k] if i in rel) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    return 1.0 if any(relevant.get(i, 0) > 0 for i in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Mapping[str, float], k: int | None = None) -> float:
    for rank, i in enumerate(retrieved[:k] if k else retrieved, start=1):
        if relevant.get(i, 0) > 0:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    gains = [relevant.get(i, 0.0) for i in retrieved[:k]]
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def citation_precision(cited: Sequence[str], relevant: Mapping[str, float]) -> float:
    """Share of cited chunks that are actually relevant to the question."""
    if not cited:
        return 0.0
    return sum(1 for c in cited if relevant.get(c, 0) > 0) / len(cited)


def citation_validity(cited: Sequence[str], context: Sequence[str]) -> float:
    """Share of citations that point at chunks that were really given to the LLM (1.0 = none fabricated)."""
    if not cited:
        return 1.0
    ctx = set(context)
    return sum(1 for c in cited if c in ctx) / len(cited)

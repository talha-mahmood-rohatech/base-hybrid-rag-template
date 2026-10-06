"""Deterministic feature-hashing embeddings.

NOT a semantic model. Intended for tests, CI and offline smoke runs: identical inputs give
identical vectors, lexical overlap gives positive cosine similarity, and any dimension
(e.g. 1024/1536/3072) can be produced without downloading anything.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence

from app.providers.embeddings.base import EmbeddingProvider

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    for suf in ("ing", "es", "ed", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


class HashingEmbeddingProvider(EmbeddingProvider):
    def _vec(self, text: str) -> list[float]:
        dim = self.spec.dimension
        vec = [0.0] * dim
        toks = [_stem(t) for t in _WORD_RE.findall(text.lower())]
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        return [self._vec(t) for t in texts]

    async def _embed_query(self, text: str) -> Sequence[float]:
        return self._vec(text)

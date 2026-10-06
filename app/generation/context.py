"""Build the LLM context from reranked candidates.

* near-duplicate removal (exact checksum + word-shingle Jaccard)
* token budget (greedy in relevance order; oversize chunks are skipped, not truncated)
* sensible ordering: ``grouped`` keeps chunks of one document together (documents ordered by
  their best rank, chunks by position) so the model reads coherent passages; ``relevance``
  keeps pure rerank order
* stable citation ids ``[1]..[n]`` with source metadata in each block header
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.tokens import TokenCounter
from app.retrieval.types import Candidate

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return frozenset({tuple(words)}) if words else frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(slots=True)
class ContextBlock:
    citation_id: int
    candidate: Candidate
    text: str
    tokens: int

    def source_label(self) -> str:
        c = self.candidate.chunk
        assert c is not None
        parts = [f"Document: {c.document_name}"]
        if c.section:
            parts.append(f"Section: {c.section}")
        if c.page is not None:
            parts.append(
                f"Page: {c.page}" if c.page_end in (None, c.page) else f"Pages: {c.page}-{c.page_end}"
            )
        return " | ".join(parts)


@dataclass(slots=True)
class BuiltContext:
    blocks: list[ContextBlock]
    text: str
    token_count: int
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def by_citation(self) -> dict[int, ContextBlock]:
        return {b.citation_id: b for b in self.blocks}


class ContextBuilder:
    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        max_tokens: int = 6000,
        dedup_threshold: float = 0.85,
        ordering: Literal["grouped", "relevance"] = "grouped",
    ) -> None:
        self.tokens = token_counter
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self.ordering = ordering

    def build(self, candidates: Sequence[Candidate]) -> BuiltContext:
        selected: list[tuple[Candidate, int]] = []
        dropped: list[dict[str, Any]] = []
        seen_checksums: set[str] = set()
        seen_shingles: list[tuple[Candidate, frozenset]] = []
        used = 0

        for cand in candidates:
            chunk = cand.chunk
            if chunk is None:
                continue
            if chunk.checksum in seen_checksums:
                dropped.append({"chunk_id": str(cand.chunk_id), "reason": "duplicate_exact"})
                continue
            sh = shingles(chunk.text)
            dup_of = next(
                (other for other, osh in seen_shingles if jaccard(sh, osh) >= self.dedup_threshold), None
            )
            if dup_of is not None:
                dropped.append(
                    {"chunk_id": str(cand.chunk_id), "reason": "near_duplicate", "of": str(dup_of.chunk_id)}
                )
                continue
            # Header overhead (~20 tokens) is budgeted with the chunk.
            cost = chunk.token_count + 20
            if used + cost > self.max_tokens:
                dropped.append({"chunk_id": str(cand.chunk_id), "reason": "token_budget", "tokens": cost})
                continue
            used += cost
            seen_checksums.add(chunk.checksum)
            seen_shingles.append((cand, sh))
            selected.append((cand, cost))

        if self.ordering == "grouped":
            doc_best: dict = {}
            for i, (cand, _) in enumerate(selected):
                doc_best.setdefault(cand.document_id, i)
            selected.sort(key=lambda t: (doc_best[t[0].document_id], t[0].chunk.chunk_index))  # type: ignore[union-attr]

        blocks: list[ContextBlock] = []
        parts: list[str] = []
        for cid, (cand, cost) in enumerate(selected, start=1):
            block = ContextBlock(citation_id=cid, candidate=cand, text=cand.chunk.text.strip(), tokens=cost)  # type: ignore[union-attr]
            blocks.append(block)
            parts.append(f"[{cid}] {block.source_label()}\n{block.text}")
        text = "\n\n".join(parts)
        return BuiltContext(blocks=blocks, text=text, token_count=self.tokens.count(text), dropped=dropped)

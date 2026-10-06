"""Citation extraction and validation.

The model cites passages as ``[n]``. Only ids that exist in the built context are accepted;
any other marker is removed from the answer and recorded, so fabricated citations never
reach the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.generation.context import BuiltContext

_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
# Some models (e.g. gpt-oss) cite with fullwidth brackets: 【7】, 【7, 8】 or 【7†source】.
_FULLWIDTH_CITE_RE = re.compile(r"【\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*(?:†[^】]*)?】")


@dataclass(slots=True)
class Citation:
    citation_id: int
    document_id: str
    document_name: str
    document_version_id: str
    chunk_id: str
    page: int | None
    page_end: int | None
    section: str | None
    source: str | None
    snippet: str
    reranker_score: float | None
    rrf_score: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class CitationResult:
    answer: str
    citations: list[Citation]
    invalid_ids: list[int] = field(default_factory=list)


def extract_citations(answer: str, context: BuiltContext) -> CitationResult:
    blocks = context.by_citation()
    order: list[int] = []
    invalid: list[int] = []

    def _replace(m: re.Match) -> str:
        ids = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = []
        for i in ids:
            if i in blocks:
                valid.append(i)
                if i not in order:
                    order.append(i)
            elif i not in invalid:
                invalid.append(i)
        return "".join(f"[{i}]" for i in valid)

    normalized = _FULLWIDTH_CITE_RE.sub(lambda m: f"[{m.group(1)}]", answer)
    cleaned = _CITE_RE.sub(_replace, normalized)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations: list[Citation] = []
    for cid in order:
        block = blocks[cid]
        chunk = block.candidate.chunk
        assert chunk is not None
        citations.append(
            Citation(
                citation_id=cid,
                document_id=str(chunk.document_id),
                document_name=chunk.document_name,
                document_version_id=str(chunk.document_version_id),
                chunk_id=str(chunk.chunk_id),
                page=chunk.page,
                page_end=chunk.page_end,
                section=chunk.section,
                source=chunk.source,
                snippet=chunk.text[:300],
                reranker_score=block.candidate.reranker_score,
                rrf_score=block.candidate.fused_score,
            )
        )
    return CitationResult(answer=cleaned, citations=citations, invalid_ids=invalid)

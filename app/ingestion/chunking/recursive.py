"""Recursive, structure-aware chunker.

1. Consecutive segments with the same section (heading path) form a *group*. Chunks never
   cross group boundaries, so every chunk has exactly one section; chunks may span pages.
2. Each group is split recursively on progressively finer separators (paragraph, line,
   sentence, clause, word, character) into atomic pieces no larger than ``chunk_size``.
   Separators stay attached to the preceding piece, so pieces are exact substrings and
   character offsets are preserved.
3. Pieces are merged greedily up to ``chunk_size`` with ``chunk_overlap`` carried over.

Sizes are measured in tokens (default) or characters.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from app.core.tokens import TokenCounter
from app.ingestion.chunking.base import ChunkDraft, Chunker
from app.ingestion.loaders.base import ParsedDocument

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")
_HAS_WORD = re.compile(r"\w", re.UNICODE)


class RecursiveStructureChunker(Chunker):
    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        unit: str = "tokens",
        min_chunk_chars: int = 10,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.counter = token_counter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unit = unit
        self.min_chunk_chars = min_chunk_chars
        self.separators = tuple(separators)
        self._len: Callable[[str], int] = token_counter.count if unit == "tokens" else len

    # -- public -----------------------------------------------------------------------
    def chunk(self, document: ParsedDocument) -> list[ChunkDraft]:
        # Global offsets follow ParsedDocument.text == "\n\n".join(segment texts).
        groups: list[tuple[str | None, int, int, list[tuple[int, int | None]]]] = []
        offset = 0
        for i, seg in enumerate(document.segments):
            if i > 0:
                offset += 2
            start, end = offset, offset + len(seg.text)
            if groups and groups[-1][0] == seg.section:
                section, gstart, _, pages = groups[-1]
                pages.append((start, seg.page))
                groups[-1] = (section, gstart, end, pages)
            else:
                groups.append((seg.section, start, end, [(start, seg.page)]))
            offset = end

        full = document.text
        drafts: list[ChunkDraft] = []
        for section, gstart, gend, pages in groups:
            pieces = self._split(full, gstart, gend, self.separators)
            for cstart, cend in self._merge(full, pieces):
                # Trim whitespace while keeping offsets exact.
                raw = full[cstart:cend]
                lstrip = len(raw) - len(raw.lstrip())
                rstrip = len(raw) - len(raw.rstrip())
                cstart, cend = cstart + lstrip, cend - rstrip
                text = full[cstart:cend]
                if len(text) < self.min_chunk_chars or not _HAS_WORD.search(text):
                    continue
                drafts.append(
                    ChunkDraft(
                        index=len(drafts),
                        text=text,
                        char_start=cstart,
                        char_end=cend,
                        token_count=self.counter.count(text),
                        page=_page_at(pages, cstart),
                        page_end=_page_at(pages, max(cstart, cend - 1)),
                        section=section,
                    )
                )
        return drafts

    # -- internals --------------------------------------------------------------------
    def _split(self, text: str, start: int, end: int, seps: Sequence[str]) -> list[tuple[int, int]]:
        if start >= end:
            return []
        if self._len(text[start:end]) <= self.chunk_size:
            return [(start, end)]
        for i, sep in enumerate(seps):
            if sep == "":
                return self._hard_split(text, start, end)
            if text.find(sep, start, end) != -1:
                rest = seps[i + 1 :]
                break
        else:
            return self._hard_split(text, start, end)

        out: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            idx = text.find(sep, pos, end)
            pend = end if idx == -1 else idx + len(sep)
            if self._len(text[pos:pend]) <= self.chunk_size:
                out.append((pos, pend))
            else:
                out.extend(self._split(text, pos, pend, rest))
            pos = pend
        return out

    def _hard_split(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        out = []
        pos = start
        while pos < end:
            window = self.chunk_size if self.unit == "characters" else self.chunk_size * 4
            stop = min(end, pos + window)
            while stop - pos > 1 and self._len(text[pos:stop]) > self.chunk_size:
                stop = pos + max(1, (stop - pos) * 3 // 4)
            out.append((pos, stop))
            pos = stop
        return out

    def _merge(self, text: str, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
        chunks: list[tuple[int, int]] = []
        cur: list[tuple[int, int, int]] = []
        cur_len = 0
        for s, e in pieces:
            plen = self._len(text[s:e])
            if cur and cur_len + plen > self.chunk_size:
                chunks.append((cur[0][0], cur[-1][1]))
                while cur and (cur_len > self.chunk_overlap or cur_len + plen > self.chunk_size):
                    cur_len -= cur.pop(0)[2]
            cur.append((s, e, plen))
            cur_len += plen
        if cur:
            chunks.append((cur[0][0], cur[-1][1]))
        return chunks


def _page_at(pages: list[tuple[int, int | None]], pos: int) -> int | None:
    page = pages[0][1]
    for start, p in pages:
        if start <= pos:
            page = p
        else:
            break
    return page

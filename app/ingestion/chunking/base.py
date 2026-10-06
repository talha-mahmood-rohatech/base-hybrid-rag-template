from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.ingestion.loaders.base import ParsedDocument


@dataclass(slots=True)
class ChunkDraft:
    index: int
    text: str
    char_start: int  # offsets into ParsedDocument.text
    char_end: int
    token_count: int
    page: int | None = None
    page_end: int | None = None
    section: str | None = None


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[ChunkDraft]: ...

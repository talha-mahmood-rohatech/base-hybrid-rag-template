from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Segment:
    """A structural unit of a document: a page (PDF) or a section body (MD/HTML/DOCX)."""

    text: str
    page: int | None = None
    section: str | None = None  # heading path, e.g. "Returns > Refunds"


@dataclass(slots=True)
class ParsedDocument:
    segments: list[Segment]
    title: str | None = None
    page_count: int | None = None
    parser: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


class DocumentLoader(ABC):
    """Parses raw bytes of one document type into structured segments."""

    document_type: str = ""
    extensions: tuple[str, ...] = ()
    mime_types: tuple[str, ...] = ()

    @abstractmethod
    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument: ...


def decode_text(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    for enc in ("utf-8", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace")


class HeadingStack:
    """Tracks the current heading path for structure-aware sectioning."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title.strip()))

    @property
    def path(self) -> str | None:
        return " > ".join(t for _, t in self._stack if t) or None

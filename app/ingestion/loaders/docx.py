"""DOCX loader: paragraphs + tables in document order, sectioned by Heading/Title styles."""

from __future__ import annotations

import io
import re

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import DocumentLoader, HeadingStack, ParsedDocument, Segment

_HEADING_RE = re.compile(r"^Heading (\d)$", re.I)


class DocxLoader(DocumentLoader):
    document_type = "docx"
    extensions = (".docx",)
    mime_types = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",)

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise DocumentParseError(f"Failed to parse DOCX: {exc}") from exc

        stack = HeadingStack()
        segments: list[Segment] = []
        buf: list[str] = []
        title = (document.core_properties.title or "").strip() or None

        def flush() -> None:
            if buf:
                segments.append(Segment(text="\n".join(buf), section=stack.path))
                buf.clear()

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style = (item.style.name if item.style is not None else "") or ""
                m = _HEADING_RE.match(style)
                if m or style == "Title":
                    flush()
                    level = int(m.group(1)) if m else 0
                    stack.push(level, text)
                    if title is None and level <= 1:
                        title = text
                    continue
                prefix = "- " if "List" in style else ""
                buf.append(prefix + text)
            elif isinstance(item, Table):
                for row in item.rows:
                    cells: list[str] = []
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t and (not cells or cells[-1] != t):  # merged cells repeat
                            cells.append(t)
                    if cells:
                        buf.append(" | ".join(cells))
        flush()
        if not segments:
            raise DocumentParseError("DOCX contains no text")
        return ParsedDocument(segments=segments, title=title, parser="python-docx")

"""Markdown loader: splits into sections by ATX/Setext headings (code fences respected)."""

from __future__ import annotations

from markdown_it import MarkdownIt

from app.ingestion.loaders.base import DocumentLoader, HeadingStack, ParsedDocument, Segment, decode_text


class MarkdownLoader(DocumentLoader):
    document_type = "md"
    extensions = (".md", ".markdown", ".mdx")
    mime_types = ("text/markdown", "text/x-markdown")

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark")

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        text = decode_text(data)
        lines = text.splitlines()
        tokens = self._md.parse(text)

        headings: list[tuple[int, int, int, str]] = []  # (start_line, end_line, level, title)
        for i, tok in enumerate(tokens):
            if tok.type == "heading_open" and tok.map:
                heading = tokens[i + 1].content if i + 1 < len(tokens) else ""
                headings.append((tok.map[0], tok.map[1], int(tok.tag[1]), heading))

        stack = HeadingStack()
        segments: list[Segment] = []
        title: str | None = None

        def emit(start: int, end: int) -> None:
            body = "\n".join(lines[start:end]).strip()
            if body:
                segments.append(Segment(text=body, section=stack.path))

        cursor = 0
        for start, end, level, htitle in headings:
            emit(cursor, start)
            stack.push(level, htitle)
            if title is None and level == 1:
                title = htitle
            cursor = end
        emit(cursor, len(lines))
        if not segments and text.strip():
            segments.append(Segment(text=text.strip()))
        return ParsedDocument(segments=segments, title=title, parser="markdown-it")

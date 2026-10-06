from __future__ import annotations

from app.ingestion.loaders.base import DocumentLoader, ParsedDocument, Segment, decode_text


class TextLoader(DocumentLoader):
    document_type = "txt"
    extensions = (".txt", ".text", ".log", ".csv")
    mime_types = ("text/plain", "text/csv")

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        text = decode_text(data)
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), None)
        title = first[:200] if first and len(first) < 120 else None
        return ParsedDocument(segments=[Segment(text=text)], title=title, parser="text")

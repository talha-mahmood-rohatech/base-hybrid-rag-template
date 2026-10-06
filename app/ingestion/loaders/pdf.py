"""PDF loader: one segment per page (1-based page numbers preserved for citations)."""

from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import DocumentLoader, ParsedDocument, Segment


class PDFLoader(DocumentLoader):
    document_type = "pdf"
    extensions = (".pdf",)
    mime_types = ("application/pdf",)

    def load(self, data: bytes, filename: str | None = None) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                reader.decrypt("")
            segments = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    segments.append(Segment(text=text, page=i))
            meta: Any = reader.metadata or {}
            title = (meta.get("/Title") or "").strip() or None
            page_count = len(reader.pages)
        except Exception as exc:
            raise DocumentParseError(f"Failed to parse PDF: {exc}") from exc
        if not segments:
            raise DocumentParseError(
                "PDF contains no extractable text (scanned PDFs need OCR, not supported in V1)"
            )
        return ParsedDocument(segments=segments, title=title, page_count=page_count, parser="pypdf")

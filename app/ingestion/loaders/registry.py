from __future__ import annotations

from pathlib import PurePath

from app.core.errors import UnsupportedDocumentError
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.docx import DocxLoader
from app.ingestion.loaders.html import HTMLLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.text import TextLoader


class LoaderRegistry:
    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders = loaders or [PDFLoader(), DocxLoader(), MarkdownLoader(), HTMLLoader(), TextLoader()]
        self._by_type = {loader.document_type: loader for loader in self._loaders}

    @property
    def supported_types(self) -> list[str]:
        return sorted(self._by_type)

    def detect_type(
        self, filename: str | None, mime_type: str | None = None, data: bytes | None = None
    ) -> str:
        if filename:
            ext = PurePath(filename).suffix.lower()
            for loader in self._loaders:
                if ext in loader.extensions:
                    return loader.document_type
        if mime_type:
            base = mime_type.split(";")[0].strip().lower()
            for loader in self._loaders:
                if base in loader.mime_types:
                    return loader.document_type
        if data is not None:
            if data.startswith(b"%PDF"):
                return "pdf"
            if data.startswith(b"PK\x03\x04") and b"word/" in data[:4096]:
                return "docx"
            head = data[:512].lstrip().lower()
            if head.startswith((b"<!doctype html", b"<html")):
                return "html"
        raise UnsupportedDocumentError(
            f"Cannot determine a supported document type for '{filename}' ({mime_type}). "
            f"Supported: {self.supported_types}"
        )

    def get(self, document_type: str) -> DocumentLoader:
        try:
            return self._by_type[document_type]
        except KeyError as exc:
            raise UnsupportedDocumentError(
                f"Unsupported document type '{document_type}'. Supported: {self.supported_types}"
            ) from exc

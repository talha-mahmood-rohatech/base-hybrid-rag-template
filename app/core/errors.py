"""Domain errors. The API layer maps these to HTTP responses."""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all platform errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(RAGError):
    code = "not_found"
    status_code = 404


class ConflictError(RAGError):
    code = "conflict"
    status_code = 409


class ValidationError(RAGError):
    code = "validation_error"
    status_code = 422


class AuthenticationError(RAGError):
    code = "unauthorized"
    status_code = 401


class UnsupportedDocumentError(ValidationError):
    code = "unsupported_document"


class DocumentParseError(RAGError):
    code = "document_parse_error"
    status_code = 422


class VectorDimensionMismatchError(RAGError):
    """A vector does not match the dimension of its embedding configuration/collection."""

    code = "vector_dimension_mismatch"
    status_code = 500


class EmbeddingConfigMismatchError(RAGError):
    """An existing Qdrant collection is incompatible with its embedding configuration."""

    code = "embedding_config_mismatch"
    status_code = 500


class ProviderError(RAGError):
    """A model provider (embedding, reranker, LLM) or backing store failed."""

    code = "provider_error"
    status_code = 502


class ProviderConfigurationError(RAGError):
    code = "provider_configuration_error"
    status_code = 500

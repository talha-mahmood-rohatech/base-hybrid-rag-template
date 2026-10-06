"""Application configuration.

Settings are resolved in this order (highest priority first):

1. Explicit init kwargs (tests)
2. Environment variables (nested with ``__``, e.g. ``EMBEDDING__MODEL``)
3. ``.env`` file
4. YAML config file (``RAG_CONFIG_FILE``, default ``config/rag.yaml``)
5. Defaults declared below
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DistanceMetric = Literal["cosine", "dot", "euclid"]


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    pool_size: int = 10
    max_overflow: int = 10
    echo: bool = False


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection_prefix: str = "rag_chunks"
    timeout_s: float = 30.0
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    search_hnsw_ef: int | None = 128
    on_disk_vectors: bool = False
    upsert_batch_size: int = 128


class EmbeddingSettings(BaseModel):
    """Default embedding configuration for *new* knowledge bases.

    Existing knowledge bases stay pinned to the embedding configuration they were
    created with (tracked in the ``embedding_configs`` table), so changing these values
    never silently mixes vector spaces.
    """

    provider: Literal["fastembed", "openai", "hashing"] = "fastembed"
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    dimension: int = Field(default=1024, gt=0, le=8192)
    version: str = "v1"
    distance_metric: DistanceMetric = "cosine"
    batch_size: int = 16
    max_input_tokens: int | None = None
    # Instruction prefixes; ``None`` means "use the known default for this model".
    query_prefix: str | None = None
    document_prefix: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_s: float = 60.0


class ChunkingSettings(BaseModel):
    strategy: Literal["recursive"] = "recursive"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    unit: Literal["tokens", "characters"] = "tokens"
    # Chunks shorter than this many characters (e.g. stray page numbers) are dropped.
    min_chunk_chars: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalSettings(BaseModel):
    dense_top_k: int = Field(default=50, ge=1, le=1000)
    sparse_top_k: int = Field(default=50, ge=1, le=1000)
    fusion: Literal["rrf", "relative_score"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    rrf_top_k: int = Field(default=40, ge=1, le=1000)
    final_top_k: int = Field(default=8, ge=1, le=100)
    sparse_provider: Literal["postgres_bm25", "postgres_fts"] = "postgres_bm25"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    fts_language: str = "english"
    # If one retriever fails, continue with the other and mark the trace as degraded.
    tolerate_retriever_failure: bool = True


class RerankerSettings(BaseModel):
    provider: Literal["fastembed", "cohere", "none"] = "fastembed"
    model: str = "jinaai/jina-reranker-v2-base-multilingual"
    batch_size: int = 16
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_s: float = 60.0
    # Include section/heading path in the text the cross-encoder sees.
    include_section: bool = True


class LLMSettings(BaseModel):
    provider: Literal["openai_compatible", "anthropic", "extractive"] = "openai_compatible"
    model: str = "qwen2.5:3b"
    base_url: str | None = "http://localhost:11434/v1"
    api_key: SecretStr | None = None
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: float = 180.0
    # Anthropic only: output_config.effort and server-side refusal fallbacks.
    effort: str | None = "medium"
    use_fallbacks: bool = True


class ContextSettings(BaseModel):
    max_context_tokens: int = Field(default=6000, gt=0)
    dedup_threshold: float = Field(default=0.85, gt=0, le=1.0)
    ordering: Literal["grouped", "relevance"] = "grouped"


class StorageSettings(BaseModel):
    blob_dir: str = "./data/blobs"
    max_upload_bytes: int = 50 * 1024 * 1024


class WorkerSettings(BaseModel):
    embedded: bool = True
    poll_interval_s: float = 1.0
    max_attempts: int = 3
    lease_timeout_s: int = 900


class SecuritySettings(BaseModel):
    admin_api_key: SecretStr = SecretStr("change-me-admin-key")


class TokenizerSettings(BaseModel):
    encoding: str = "cl100k_base"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "hybrid-rag"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    database: DatabaseSettings = DatabaseSettings()
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    reranker: RerankerSettings = RerankerSettings()
    llm: LLMSettings = LLMSettings()
    context: ContextSettings = ContextSettings()
    storage: StorageSettings = StorageSettings()
    worker: WorkerSettings = WorkerSettings()
    security: SecuritySettings = SecuritySettings()
    tokenizer: TokenizerSettings = TokenizerSettings()

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _no_default_secrets_in_prod(self) -> Settings:
        if self.environment in ("staging", "production") and (
            self.security.admin_api_key.get_secret_value()
            == SecuritySettings().admin_api_key.get_secret_value()
        ):
            raise ValueError("SECURITY__ADMIN_API_KEY must be changed outside local/test environments")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        config_file = Path(os.environ.get("RAG_CONFIG_FILE", "config/rag.yaml"))
        if config_file.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_file))
        sources.append(file_secret_settings)
        return tuple(sources)

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to expose (secrets masked)."""
        return self.model_dump(mode="json", exclude={"security"})


@lru_cache
def get_settings() -> Settings:
    return Settings()

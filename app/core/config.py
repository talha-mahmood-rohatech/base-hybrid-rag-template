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

from app.voice.vad.config import VadConfig

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
    # CPU latency controls. Cross-encoder cost grows with (candidates x passage length):
    # rerank only the best N fused candidates (None = all rrf_top_k) ...
    max_candidates: int | None = Field(default=None, ge=1, le=1000)
    # ... and cap the passage text the cross-encoder reads (the LLM still gets the full chunk).
    max_chars: int | None = Field(default=None, ge=100)
    # fastembed only: use any cross-encoder with an ONNX export on Hugging Face.
    onnx_repo: str | None = None  # e.g. "onnx-community/bge-reranker-v2-m3-ONNX"
    onnx_file: str | None = None  # e.g. "onnx/model_int8.onnx"
    onnx_additional_files: list[str] = Field(default_factory=list)  # e.g. ["onnx/model.onnx_data"]


class LLMSettings(BaseModel):
    provider: Literal["openai_compatible", "groq", "anthropic", "extractive"] = "openai_compatible"
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


class VoiceSettings(BaseModel):
    """Voice pipeline (mic -> Silero VAD -> STT -> hybrid RAG -> TTS), ported from the Leap agent."""

    enabled: bool = True
    # Speech-to-text (Groq Whisper). The key falls back to LLM__API_KEY when LLM__PROVIDER=groq.
    stt_provider: Literal["groq"] = "groq"
    stt_model: str = "whisper-large-v3"
    stt_api_key: SecretStr | None = None
    stt_base_url: str | None = None
    # Force the decode language (ISO code, e.g. "en"); None = auto-detect per utterance.
    stt_language: str | None = None
    # Vocabulary hint for Whisper (domain terms, product names), e.g.
    # "Riba, Ijarah, Murabaha, Musharakah, Mudarabah, Shariah". Clients can override per session.
    stt_prompt: str | None = None
    # Reliability gate: below/above these the utterance is treated as unintelligible.
    stt_min_avg_logprob: float = -1.5
    stt_max_no_speech_prob: float = 0.85
    stt_timeout_s: float = 60.0
    # Text-to-speech (Soniox). "none" = voice answers come back as text only.
    tts_provider: Literal["soniox", "none"] = "soniox"
    tts_model: str = "tts-rt-v2"
    tts_api_key: SecretStr | None = None
    tts_base_url: str | None = None
    tts_voice: str = "Nina"
    tts_speed: float = Field(default=0.95, ge=0.7, le=1.3)
    # Language spoken when STT did not report one (ISO code).
    tts_language: str = "en"
    tts_cache_dir: str = "./data/tts-cache"
    tts_timeout_s: float = 30.0
    # Voice answers are spoken, so long answers are cut at a sentence boundary for speech
    # (the full text and citations are still sent).
    max_spoken_chars: int = Field(default=1200, ge=100)
    # Answers are voiced in sentence groups so playback starts once the first (short) group is
    # synthesized - Soniox REST synthesis runs at about real time - while the rest is voiced in
    # the background.
    tts_first_chunk_chars: int = Field(default=90, ge=20)
    tts_chunk_chars: int = Field(default=200, ge=40)
    # How long the browser asks the VAD to ignore detection after the assistant stops
    # speaking (room/speaker echo tail). Sent to clients in the session_ready message.
    echo_hold_ms: int = Field(default=500, ge=0)
    vad: VadConfig = VadConfig()


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
    voice: VoiceSettings = VoiceSettings()

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

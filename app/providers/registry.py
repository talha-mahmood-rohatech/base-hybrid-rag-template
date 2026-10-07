"""Builds provider instances from configuration. The only place that knows concrete classes."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import EmbeddingSettings, LLMSettings, RerankerSettings, RetrievalSettings, Settings
from app.core.errors import ProviderConfigurationError
from app.providers.embeddings.base import EmbeddingProvider, EmbeddingSpec
from app.providers.llms.base import LLMProvider
from app.providers.rerankers.base import PassthroughReranker, Reranker
from app.providers.sparse.base import SparseSearch
from app.providers.stt.base import SpeechToText
from app.providers.tts.base import NoTextToSpeech, TextToSpeech
from app.providers.vectorstores.qdrant import QdrantVectorStore
from app.retrieval.fusion.base import FusionStrategy
from app.retrieval.fusion.relative_score import RelativeScoreFusion
from app.retrieval.fusion.rrf import ReciprocalRankFusion


def _secret(v: Any) -> str | None:
    return v.get_secret_value() if v is not None else None


def model_cache_dir() -> str | None:
    return os.environ.get("FASTEMBED_CACHE_PATH") or None


def build_embedding_provider(
    spec: EmbeddingSpec, settings: EmbeddingSettings, params: dict | None = None
) -> EmbeddingProvider:
    """Build the provider for an embedding *config* (spec comes from the DB row)."""
    params = params or {}
    q_prefix = params.get("query_prefix", settings.query_prefix if settings.model == spec.model else None)
    d_prefix = params.get(
        "document_prefix", settings.document_prefix if settings.model == spec.model else None
    )
    if spec.provider == "fastembed":
        from app.providers.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider(
            spec,
            batch_size=settings.batch_size,
            query_prefix=q_prefix,
            document_prefix=d_prefix,
            cache_dir=model_cache_dir(),
            max_input_tokens=params.get("max_input_tokens", settings.max_input_tokens),
        )
    if spec.provider == "openai":
        from app.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            spec,
            api_key=_secret(settings.api_key),
            base_url=params.get("base_url", settings.base_url),
            batch_size=settings.batch_size,
            timeout_s=settings.timeout_s,
            query_prefix=q_prefix,
            document_prefix=d_prefix,
        )
    if spec.provider == "hashing":
        from app.providers.embeddings.hashing_provider import HashingEmbeddingProvider

        return HashingEmbeddingProvider(spec)
    raise ProviderConfigurationError(f"Unknown embedding provider '{spec.provider}'")


def build_reranker(settings: RerankerSettings) -> Reranker:
    if settings.provider == "none":
        return PassthroughReranker()
    if settings.provider == "fastembed":
        from app.providers.rerankers.fastembed_reranker import FastEmbedCrossEncoderReranker

        return FastEmbedCrossEncoderReranker(
            settings.model,
            batch_size=settings.batch_size,
            cache_dir=model_cache_dir(),
            onnx_repo=settings.onnx_repo,
            onnx_file=settings.onnx_file,
            onnx_additional_files=settings.onnx_additional_files,
        )
    if settings.provider == "cohere":
        from app.providers.rerankers.cohere_reranker import CohereReranker

        return CohereReranker(
            settings.model,
            api_key=_secret(settings.api_key),
            base_url=settings.base_url,
            timeout_s=settings.timeout_s,
        )
    raise ProviderConfigurationError(f"Unknown reranker provider '{settings.provider}'")


def build_llm(settings: LLMSettings) -> LLMProvider:
    if settings.provider == "openai_compatible":
        from app.providers.llms.openai_compatible import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            settings.model,
            base_url=settings.base_url,
            api_key=_secret(settings.api_key),
            timeout_s=settings.timeout_s,
        )
    if settings.provider == "groq":
        from app.providers.llms.openai_compatible import GroqLLM

        return GroqLLM(
            settings.model,
            api_key=_secret(settings.api_key),
            # LLM__BASE_URL defaults to the local Ollama URL; only honour it if it points at Groq.
            base_url=settings.base_url if settings.base_url and "groq" in settings.base_url else None,
            timeout_s=settings.timeout_s,
        )
    if settings.provider == "anthropic":
        from app.providers.llms.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            settings.model,
            api_key=_secret(settings.api_key),
            base_url=settings.base_url if settings.base_url and "anthropic" in settings.base_url else None,
            timeout_s=settings.timeout_s,
            effort=settings.effort,
            use_fallbacks=settings.use_fallbacks,
        )
    if settings.provider == "extractive":
        from app.providers.llms.extractive import ExtractiveLLM

        return ExtractiveLLM()
    raise ProviderConfigurationError(f"Unknown LLM provider '{settings.provider}'")


def build_sparse_search(
    settings: RetrievalSettings, session_factory: async_sessionmaker[AsyncSession]
) -> SparseSearch:
    from app.providers.sparse.postgres import PostgresBM25Search, PostgresFTSSearch

    if settings.sparse_provider == "postgres_bm25":
        return PostgresBM25Search(session_factory, k1=settings.bm25_k1, b=settings.bm25_b)
    if settings.sparse_provider == "postgres_fts":
        return PostgresFTSSearch(session_factory)
    raise ProviderConfigurationError(f"Unknown sparse provider '{settings.sparse_provider}'")


def build_fusion(name: str, rrf_k: int) -> FusionStrategy:
    if name == "rrf":
        return ReciprocalRankFusion(k=rrf_k)
    if name == "relative_score":
        return RelativeScoreFusion()
    raise ProviderConfigurationError(f"Unknown fusion strategy '{name}'")


def build_vector_store(settings: Settings) -> QdrantVectorStore:
    q = settings.qdrant
    return QdrantVectorStore.from_url(
        q.url,
        api_key=_secret(q.api_key),
        timeout_s=q.timeout_s,
        hnsw_m=q.hnsw_m,
        hnsw_ef_construct=q.hnsw_ef_construct,
        search_hnsw_ef=q.search_hnsw_ef,
        on_disk_vectors=q.on_disk_vectors,
        upsert_batch_size=q.upsert_batch_size,
    )


class EmbeddingProviderCache:
    """One provider instance per embedding configuration (models are loaded once)."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self._cache: dict[EmbeddingSpec, EmbeddingProvider] = {}

    def register(self, provider: EmbeddingProvider) -> None:
        self._cache[provider.spec] = provider

    def get(self, spec: EmbeddingSpec, params: dict | None = None) -> EmbeddingProvider:
        provider = self._cache.get(spec)
        if provider is None:
            provider = self._cache[spec] = build_embedding_provider(spec, self.settings, params)
        return provider

    async def aclose(self) -> None:
        for p in self._cache.values():
            await p.aclose()


def build_stt(settings: Settings) -> SpeechToText:
    v = settings.voice
    if v.stt_provider == "groq":
        from app.providers.stt.groq_whisper import GroqWhisperSTT

        key = _secret(v.stt_api_key)
        if not key and settings.llm.provider == "groq":
            key = _secret(settings.llm.api_key)  # one Groq key for LLM and STT
        return GroqWhisperSTT(v.stt_model, api_key=key, base_url=v.stt_base_url, timeout_s=v.stt_timeout_s)
    raise ProviderConfigurationError(f"Unknown speech-to-text provider '{v.stt_provider}'")


def build_tts(settings: Settings) -> TextToSpeech:
    v = settings.voice
    if v.tts_provider == "none":
        return NoTextToSpeech()
    if v.tts_provider == "soniox":
        from app.providers.tts.cache import CachedTTS
        from app.providers.tts.soniox import SonioxTTS

        inner = SonioxTTS(
            v.tts_model,
            api_key=_secret(v.tts_api_key),
            voice=v.tts_voice,
            speed=v.tts_speed,
            base_url=v.tts_base_url,
            timeout_s=v.tts_timeout_s,
        )
        return CachedTTS(inner, v.tts_cache_dir)
    raise ProviderConfigurationError(f"Unknown text-to-speech provider '{v.tts_provider}'")

"""Dependency container: wires settings -> providers -> services. One per process."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import ProviderConfigurationError
from app.core.tokens import TokenCounter, get_token_counter
from app.db.session import create_engine, create_session_factory
from app.generation.context import ContextBuilder
from app.ingestion.blobstore import BlobStore, LocalBlobStore
from app.ingestion.embedding_configs import EmbeddingConfigService, collection_of, spec_of
from app.ingestion.loaders.registry import LoaderRegistry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.service import DocumentService
from app.ingestion.worker import IngestionWorker
from app.observability.tracing import TraceRecorder
from app.providers.llms.base import LLMProvider
from app.providers.registry import (
    EmbeddingProviderCache,
    build_llm,
    build_reranker,
    build_sparse_search,
    build_stt,
    build_tts,
    build_vector_store,
)
from app.providers.rerankers.base import Reranker
from app.providers.sparse.base import SparseSearch
from app.providers.stt.base import SpeechToText
from app.providers.tts.base import TextToSpeech
from app.providers.vectorstores.base import VectorStore
from app.rag.orchestrator import RAGOrchestrator
from app.retrieval.reranking.stage import RerankingStage

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    vector_store: VectorStore
    embedders: EmbeddingProviderCache
    token_counter: TokenCounter
    loaders: LoaderRegistry
    blob_store: BlobStore
    sparse_search: SparseSearch
    reranker: Reranker
    llm: LLMProvider
    embedding_configs: EmbeddingConfigService
    documents: DocumentService
    pipeline: IngestionPipeline
    worker: IngestionWorker
    orchestrator: RAGOrchestrator
    # Voice pipeline; None when disabled or not configured (voice endpoints then return 503).
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None
    voice_unavailable_reason: str | None = None

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        llm: LLMProvider | None = None,
        embedders: EmbeddingProviderCache | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        initialize: bool = True,
    ) -> Container:
        engine = create_engine(settings.database)
        session_factory = create_session_factory(engine)
        vector_store = vector_store or build_vector_store(settings)
        embedders = embedders or EmbeddingProviderCache(settings.embedding)
        token_counter = get_token_counter(settings.tokenizer.encoding)
        loaders = LoaderRegistry()
        blob_store = LocalBlobStore(settings.storage.blob_dir)
        sparse = build_sparse_search(settings.retrieval, session_factory)
        reranker = reranker or build_reranker(settings.reranker)
        llm = llm or build_llm(settings.llm)
        pipeline = IngestionPipeline(
            session_factory,
            blob_store=blob_store,
            vector_store=vector_store,
            embedders=embedders,
            loaders=loaders,
            token_counter=token_counter,
            chunking=settings.chunking,
        )
        orchestrator = RAGOrchestrator(
            session_factory,
            vector_store=vector_store,
            embedders=embedders,
            sparse_search=sparse,
            reranking=RerankingStage(
                reranker,
                include_section=settings.reranker.include_section,
                max_candidates=settings.reranker.max_candidates,
                max_chars=settings.reranker.max_chars,
            ),
            context_builder=ContextBuilder(
                token_counter,
                max_tokens=settings.context.max_context_tokens,
                dedup_threshold=settings.context.dedup_threshold,
                ordering=settings.context.ordering,
            ),
            llm=llm,
            traces=TraceRecorder(session_factory),
            retrieval_settings=settings.retrieval,
            llm_max_tokens=settings.llm.max_tokens,
            llm_temperature=settings.llm.temperature,
        )
        container = cls(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            vector_store=vector_store,
            embedders=embedders,
            token_counter=token_counter,
            loaders=loaders,
            blob_store=blob_store,
            sparse_search=sparse,
            reranker=reranker,
            llm=llm,
            embedding_configs=EmbeddingConfigService(settings.qdrant.collection_prefix),
            documents=DocumentService(blob_store, loaders, max_attempts=settings.worker.max_attempts),
            pipeline=pipeline,
            worker=IngestionWorker(
                session_factory,
                pipeline,
                poll_interval_s=settings.worker.poll_interval_s,
                lease_timeout_s=settings.worker.lease_timeout_s,
            ),
            orchestrator=orchestrator,
        )
        container._build_voice(stt, tts)
        if initialize:
            await container.initialize()
        return container

    def _build_voice(self, stt: SpeechToText | None, tts: TextToSpeech | None) -> None:
        if not self.settings.voice.enabled:
            self.voice_unavailable_reason = "Voice is disabled (VOICE__ENABLED=false)"
            return
        try:
            self.stt = stt or build_stt(self.settings)
            self.tts = tts or build_tts(self.settings)
        except ProviderConfigurationError as exc:
            # Voice is optional: a missing key must not take the text API down with it.
            self.stt = self.tts = None
            self.voice_unavailable_reason = exc.message
            logger.warning("voice pipeline unavailable", extra={"reason": exc.message})

    async def initialize(self) -> None:
        """Register the default embedding config and make sure its Qdrant collection exists."""
        async with self.session_factory() as session, session.begin():
            row = await self.embedding_configs.ensure_default(session, self.settings.embedding)
        await self.vector_store.ensure_collection(collection_of(row))
        embedder = self.embedders.get(spec_of(row), row.params)
        limit = embedder.max_input_tokens
        ch = self.settings.chunking
        if limit and ch.unit == "tokens" and ch.chunk_size > limit:
            logger.warning(
                "chunk_size exceeds the embedding model's input limit; chunk tails will be truncated "
                "for dense retrieval (sparse retrieval and reranking still see full text)",
                extra={"chunk_size": ch.chunk_size, "model": row.model, "model_max_tokens": limit},
            )
        logger.info(
            "default embedding config ready",
            extra={"model": row.model, "dimension": row.dimension, "collection": row.collection_name},
        )

    async def warmup(self) -> None:
        """Load local models eagerly so the first request is not slow."""
        for obj in (*self.embedders._cache.values(), self.reranker):
            warm = getattr(obj, "warmup", None)
            if warm is not None:
                await warm()

    async def aclose(self) -> None:
        await self.embedders.aclose()
        await self.reranker.aclose()
        await self.llm.aclose()
        for voice_provider in (self.stt, self.tts):
            if voice_provider is not None:
                await voice_provider.aclose()
        await self.vector_store.aclose()
        await self.engine.dispose()

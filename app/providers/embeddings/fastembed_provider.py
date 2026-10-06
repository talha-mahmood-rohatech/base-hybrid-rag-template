"""Local ONNX embeddings via fastembed (no GPU / torch required)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.embeddings.base import EmbeddingProvider, EmbeddingSpec

# Instruction prefixes recommended by the model authors. fastembed does not add them.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "BAAI/bge-large-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "snowflake/snowflake-arctic-embed-l": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "Qwen/Qwen3-Embedding-0.6B": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
    ),
    "Qwen/Qwen3-Embedding-0.6B-Q": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
    ),
}


KNOWN_MAX_INPUT_TOKENS: dict[str, int] = {
    "mixedbread-ai/mxbai-embed-large-v1": 512,
    "BAAI/bge-large-en-v1.5": 512,
    "BAAI/bge-base-en-v1.5": 512,
    "BAAI/bge-small-en-v1.5": 512,
    "snowflake/snowflake-arctic-embed-l": 512,
    "thenlper/gte-large": 512,
    "intfloat/multilingual-e5-large": 512,
    "nomic-ai/nomic-embed-text-v1.5": 8192,
    "jinaai/jina-embeddings-v3": 8192,
    "Qwen/Qwen3-Embedding-0.6B": 32768,
    "Qwen/Qwen3-Embedding-0.6B-Q": 32768,
}


class FastEmbedProvider(EmbeddingProvider):
    def __init__(
        self,
        spec: EmbeddingSpec,
        *,
        batch_size: int = 16,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        cache_dir: str | None = None,
        max_input_tokens: int | None = None,
        threads: int | None = None,
    ) -> None:
        super().__init__(spec, max_input_tokens=max_input_tokens or KNOWN_MAX_INPUT_TOKENS.get(spec.model))
        default_q, default_d = KNOWN_PREFIXES.get(spec.model, ("", ""))
        self.query_prefix = default_q if query_prefix is None else query_prefix
        self.document_prefix = default_d if document_prefix is None else document_prefix
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._threads = threads
        self._model: Any = None
        self._load_lock = threading.Lock()
        # ONNX sessions are thread-safe, but serializing avoids CPU oversubscription.
        self._run_lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding
                    except ImportError as exc:  # pragma: no cover
                        raise ProviderConfigurationError(
                            "fastembed is not installed; install the 'local-models' extra"
                        ) from exc
                    kwargs: dict[str, Any] = {"model_name": self.spec.model}
                    if self._cache_dir:
                        kwargs["cache_dir"] = self._cache_dir
                    if self._threads:
                        kwargs["threads"] = self._threads
                    self._model = TextEmbedding(**kwargs)
        return self._model

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def _embed_sync(self, texts: list[str]) -> list[Sequence[float]]:
        model = self._load()
        return [v.tolist() for v in model.embed(texts, batch_size=self.batch_size)]

    async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        prefixed = [self.document_prefix + t for t in texts]
        try:
            async with self._run_lock:
                return await asyncio.to_thread(self._embed_sync, prefixed)
        except ProviderConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(f"fastembed embedding failed: {exc}") from exc

    async def _embed_query(self, text: str) -> Sequence[float]:
        try:
            async with self._run_lock:
                return (await asyncio.to_thread(self._embed_sync, [self.query_prefix + text]))[0]
        except ProviderConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(f"fastembed embedding failed: {exc}") from exc

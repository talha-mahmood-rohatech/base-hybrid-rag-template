"""Local cross-encoder reranking via fastembed ONNX models.

Default model: ``jinaai/jina-reranker-v2-base-multilingual`` (cross-encoder, 1K-token
context). Also supports ``BAAI/bge-reranker-base`` and the ms-marco MiniLM cross-encoders.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.rerankers.base import Reranker


class FastEmbedCrossEncoderReranker(Reranker):
    provider = "fastembed"

    def __init__(
        self, model: str, *, batch_size: int = 16, cache_dir: str | None = None, threads: int | None = None
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._threads = threads
        self._encoder: Any = None
        self._load_lock = threading.Lock()
        self._run_lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._encoder is None:
            with self._load_lock:
                if self._encoder is None:
                    try:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder
                    except ImportError as exc:  # pragma: no cover
                        raise ProviderConfigurationError(
                            "fastembed is not installed; install the 'local-models' extra"
                        ) from exc
                    kwargs: dict[str, Any] = {"model_name": self.model}
                    if self._cache_dir:
                        kwargs["cache_dir"] = self._cache_dir
                    if self._threads:
                        kwargs["threads"] = self._threads
                    self._encoder = TextCrossEncoder(**kwargs)
        return self._encoder

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._load().rerank(query, documents, batch_size=self.batch_size)]

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            async with self._run_lock:
                return await asyncio.to_thread(self._score_sync, query, list(documents))
        except ProviderConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(f"Cross-encoder reranking failed: {exc}") from exc

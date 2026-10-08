"""Local cross-encoder reranking via fastembed ONNX models.

Default model: ``jinaai/jina-reranker-v2-base-multilingual`` (cross-encoder, 1K-token
context). fastembed's built-in models (``BAAI/bge-reranker-base``, ms-marco MiniLM, ...) work
by name. Any other cross-encoder with an ONNX export on Hugging Face can be used by naming
its repo/file (``RERANKER__ONNX_REPO`` / ``RERANKER__ONNX_FILE``); a few are pre-mapped in
``KNOWN_ONNX_RERANKERS``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.rerankers.base import Reranker

# model name -> (Hugging Face repo with ONNX export, model file, additional files)
KNOWN_ONNX_RERANKERS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # XLM-RoBERTa-large based, multilingual, 8K context. int8 quantized export (~0.6 GB).
    "BAAI/bge-reranker-v2-m3": ("onnx-community/bge-reranker-v2-m3-ONNX", "onnx/model_int8.onnx", ()),
}

_registered: set[str] = set()
_register_lock = threading.Lock()


def register_custom_model(
    model: str, repo: str, model_file: str, additional_files: Sequence[str] = ()
) -> None:
    """Teach fastembed about a cross-encoder it does not ship (idempotent, process-wide)."""
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    with _register_lock:
        if model in _registered:
            return
        if any(m["model"] == model for m in TextCrossEncoder.list_supported_models()):
            _registered.add(model)
            return
        TextCrossEncoder.add_custom_model(
            model,
            sources=ModelSource(hf=repo),
            model_file=model_file,
            additional_files=list(additional_files) or None,
        )
        _registered.add(model)


class FastEmbedCrossEncoderReranker(Reranker):
    provider = "fastembed"

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 16,
        cache_dir: str | None = None,
        threads: int | None = None,
        onnx_repo: str | None = None,
        onnx_file: str | None = None,
        onnx_additional_files: Sequence[str] = (),
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._cache_dir = cache_dir
        self._threads = threads
        known = KNOWN_ONNX_RERANKERS.get(model)
        if onnx_repo:
            self._custom: tuple[str, str, tuple[str, ...]] | None = (
                onnx_repo,
                onnx_file or "onnx/model.onnx",
                tuple(onnx_additional_files),
            )
        else:
            self._custom = known
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
                    supported = {m["model"] for m in TextCrossEncoder.list_supported_models()}
                    if self.model not in supported:
                        if self._custom is None:
                            raise ProviderConfigurationError(
                                f"Reranker model '{self.model}' is not built into fastembed; set "
                                "RERANKER__ONNX_REPO (and RERANKER__ONNX_FILE) to a Hugging Face ONNX export"
                            )
                        register_custom_model(self.model, *self._custom)
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

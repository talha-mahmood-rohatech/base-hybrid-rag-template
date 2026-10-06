"""Hybrid retrieval: run dense and sparse concurrently, then fuse."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.dense import DenseRetrieval, DenseRetriever
from app.retrieval.fusion.base import FusionStrategy
from app.retrieval.sparse import SparseRetriever
from app.retrieval.types import Candidate, RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridResult:
    dense: list[RetrievalResult]
    sparse: list[RetrievalResult]
    fused: list[Candidate]
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.errors)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        fusion: FusionStrategy,
        *,
        tolerate_failure: bool = True,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.tolerate_failure = tolerate_failure

    async def retrieve(
        self,
        dense_query: RetrievalQuery,
        sparse_query: RetrievalQuery,
        fused_top_k: int,
        *,
        use_dense: bool = True,
        use_sparse: bool = True,
    ) -> HybridResult:
        timings: dict[str, float] = {}

        async def timed_sparse() -> list[RetrievalResult]:
            t = time.perf_counter()
            try:
                return await self.sparse.retrieve(sparse_query)
            finally:
                timings["sparse_ms"] = round((time.perf_counter() - t) * 1000, 2)

        async def no_dense() -> DenseRetrieval:
            return DenseRetrieval([], 0.0, 0.0)

        async def no_sparse() -> list[RetrievalResult]:
            return []

        t0 = time.perf_counter()
        dense_out, sparse_out = await asyncio.gather(
            self.dense.retrieve(dense_query) if use_dense else no_dense(),
            timed_sparse() if use_sparse else no_sparse(),
            return_exceptions=True,
        )
        timings["retrieval_wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        errors: list[dict[str, Any]] = []
        dense_results: list[RetrievalResult] = []
        sparse_results: list[RetrievalResult] = []
        if isinstance(dense_out, BaseException):
            errors.append({"stage": "dense", "error": repr(dense_out)})
        else:
            dense_results = dense_out.results
            timings["embed_query_ms"] = dense_out.embed_ms
            timings["dense_search_ms"] = dense_out.search_ms
        if isinstance(sparse_out, BaseException):
            errors.append({"stage": "sparse", "error": repr(sparse_out)})
        else:
            sparse_results = sparse_out

        if errors:
            for e in errors:
                logger.warning("retriever failed", extra=e)
            if not self.tolerate_failure or len(errors) == 2:
                raise next(o for o in (dense_out, sparse_out) if isinstance(o, BaseException))

        t1 = time.perf_counter()
        fused = self.fusion.fuse({"dense": dense_results, "sparse": sparse_results}, top_k=fused_top_k)
        timings["fusion_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        return HybridResult(dense_results, sparse_results, fused, timings, errors)

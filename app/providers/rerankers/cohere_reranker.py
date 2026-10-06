"""Cohere Rerank API (``rerank-v3.5`` etc.). Also works with API-compatible gateways."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.rerankers.base import Reranker


class CohereReranker(Reranker):
    provider = "cohere"

    def __init__(
        self, model: str, *, api_key: str | None, base_url: str | None = None, timeout_s: float = 60.0
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError("RERANKER__API_KEY is required for the Cohere reranker")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=(base_url or "https://api.cohere.com").rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            resp = await self._client.post(
                "/v2/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": len(documents),
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cohere rerank failed: {exc}") from exc
        scores = [0.0] * len(documents)
        for item in resp.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores

    async def aclose(self) -> None:
        await self._client.aclose()

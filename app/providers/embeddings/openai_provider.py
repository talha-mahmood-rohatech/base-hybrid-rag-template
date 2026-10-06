"""OpenAI-compatible ``/embeddings`` endpoint (OpenAI, Azure-compatible gateways, vLLM, Ollama).

For OpenAI ``text-embedding-3-*`` the configured dimension is sent as ``dimensions`` so
1536/3072 (or any supported size) can be selected without code changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.embeddings.base import EmbeddingProvider, EmbeddingSpec


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        spec: EmbeddingSpec,
        *,
        api_key: str | None,
        base_url: str | None = None,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        send_dimensions: bool | None = None,
        max_input_tokens: int | None = 8191,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        super().__init__(spec, max_input_tokens=max_input_tokens)
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        if "api.openai.com" in base and not api_key:
            raise ProviderConfigurationError("EMBEDDING__API_KEY is required for the OpenAI provider")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=timeout_s)
        self.batch_size = batch_size
        self.send_dimensions = (
            spec.model.startswith("text-embedding-3") if send_dimensions is None else send_dimensions
        )
        self.query_prefix = query_prefix or ""
        self.document_prefix = document_prefix or ""

    async def _call(self, inputs: list[str]) -> list[list[float]]:
        body: dict = {"model": self.spec.model, "input": inputs, "encoding_format": "float"}
        if self.send_dimensions:
            body["dimensions"] = self.spec.dimension
        try:
            resp = await self._client.post("/embeddings", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Embedding request failed: {exc}") from exc
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    async def _embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        out: list[Sequence[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(await self._call([self.document_prefix + t for t in texts[i : i + self.batch_size]]))
        return out

    async def _embed_query(self, text: str) -> Sequence[float]:
        return (await self._call([self.query_prefix + text]))[0]

    async def aclose(self) -> None:
        await self._client.aclose()

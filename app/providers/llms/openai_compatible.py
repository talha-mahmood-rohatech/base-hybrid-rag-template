"""OpenAI-compatible ``/chat/completions`` (OpenAI, Azure gateways, vLLM, Ollama, LM Studio...)."""

from __future__ import annotations

import time

import httpx

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.llms.base import ChatMessage, LLMProvider, LLMResponse, LLMUsage


class OpenAICompatibleLLM(LLMProvider):
    provider = "openai_compatible"

    def __init__(
        self, model: str, *, base_url: str | None, api_key: str | None = None, timeout_s: float = 180.0
    ) -> None:
        super().__init__(model)
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        if "api.openai.com" in base and not api_key:
            raise ProviderConfigurationError("LLM__API_KEY is required for api.openai.com")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=timeout_s)

    async def generate(
        self, messages: list[ChatMessage], *, max_tokens: int, temperature: float = 0.0
    ) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        start = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"LLM request failed ({exc.response.status_code}): {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"LLM request failed: {exc!r}") from exc
        latency = (time.perf_counter() - start) * 1000
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice.get("message") or {}).get("content") or "",
            model=data.get("model", self.model),
            provider=self.provider,
            usage=LLMUsage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_ms=round(latency, 2),
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

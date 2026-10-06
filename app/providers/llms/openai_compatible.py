"""OpenAI-compatible ``/chat/completions`` (OpenAI, Groq, Azure gateways, vLLM, Ollama, LM Studio...)."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.llms.base import ChatMessage, LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRY_WAIT_S = 30.0


class OpenAICompatibleLLM(LLMProvider):
    provider = "openai_compatible"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model)
        base = (base_url or self.default_base_url).rstrip("/")
        if "api.openai.com" in base and not api_key:
            raise ProviderConfigurationError("LLM__API_KEY is required for api.openai.com")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=timeout_s)
        self.max_retries = max_retries

    async def _post_with_retries(self, body: dict) -> httpx.Response:
        """POST with exponential backoff on rate limits / transient errors (honours Retry-After)."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=body)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if attempt == self.max_retries:
                    raise ProviderError(f"LLM request failed: {exc!r}") from exc
                wait = min(MAX_RETRY_WAIT_S, 2.0**attempt)
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise ProviderError(f"LLM request failed ({resp.status_code}): {resp.text[:500]}")
                try:
                    wait = float(resp.headers.get("retry-after", 2.0**attempt))
                except ValueError:
                    wait = 2.0**attempt
                wait = min(MAX_RETRY_WAIT_S, max(0.5, wait))
            logger.warning(
                "LLM request retry", extra={"provider": self.provider, "attempt": attempt + 1, "wait_s": wait}
            )
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

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
            resp = await self._post_with_retries(body)
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


class GroqLLM(OpenAICompatibleLLM):
    """Groq Cloud (OpenAI-compatible API). Requires LLM__API_KEY and an explicit LLM__MODEL."""

    provider = "groq"
    default_base_url = "https://api.groq.com/openai/v1"

    def __init__(self, model: str, *, api_key: str | None, base_url: str | None = None, **kw) -> None:
        if not api_key:
            raise ProviderConfigurationError(
                "LLM__API_KEY (a Groq API key, gsk_...) is required for LLM__PROVIDER=groq"
            )
        super().__init__(model, base_url=base_url, api_key=api_key, **kw)

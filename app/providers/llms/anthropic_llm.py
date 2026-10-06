"""Anthropic Claude via the official ``anthropic`` SDK.

Notes for current models (Claude Opus 5.5 / Sonnet 5.5 / Fable 5.x):
* sampling parameters (``temperature``) are rejected, so they are only sent to older models;
* thinking depth is controlled with ``output_config.effort``;
* server-side refusal fallbacks are enabled by default (``fallbacks="default"``).
"""

from __future__ import annotations

import time
from typing import Any

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.llms.base import ChatMessage, LLMProvider, LLMResponse, LLMUsage

DEFAULT_MODEL = "claude-opus-5-5"
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable",
    "claude-mythos",
    "claude-opus-4-7",
    "claude-opus-4-8",
)
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLMProvider):
    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        effort: str | None = "medium",
        use_fallbacks: bool = True,
    ) -> None:
        super().__init__(model or DEFAULT_MODEL)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigurationError(
                "Install the 'anthropic' extra to use the Anthropic provider"
            ) from exc
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"timeout": timeout_s}
        if api_key:
            kwargs["api_key"] = api_key  # otherwise resolved from ANTHROPIC_API_KEY / ant profile
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self.effort = effort
        self.use_fallbacks = use_fallbacks and self.model.startswith(_FALLBACK_MODELS)

    async def generate(
        self, messages: list[ChatMessage], *, max_tokens: int, temperature: float = 0.0
    ) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        params: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": convo}
        if system:
            params["system"] = system
        if not self.model.startswith(_NO_SAMPLING_PREFIXES):
            params["temperature"] = temperature
        if self.effort:
            params["output_config"] = {"effort": self.effort}

        a = self._anthropic
        start = time.perf_counter()
        try:
            if self.use_fallbacks:
                resp = await self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                resp = await self._client.messages.create(**params)
        except a.RateLimitError as exc:
            raise ProviderError(f"Anthropic rate limited: {exc.message}") from exc
        except a.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise ProviderError(f"Anthropic connection error: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000

        if resp.stop_reason == "refusal":
            text = "I can't help with that request."
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text=text,
            model=resp.model,
            provider=self.provider,
            usage=LLMUsage(resp.usage.input_tokens, resp.usage.output_tokens),
            latency_ms=round(latency, 2),
            finish_reason=resp.stop_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()

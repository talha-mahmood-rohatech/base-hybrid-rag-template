import httpx
import pytest

from app.core.config import LLMSettings
from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.llms import openai_compatible
from app.providers.llms.base import ChatMessage
from app.providers.llms.openai_compatible import GroqLLM, OpenAICompatibleLLM
from app.providers.registry import build_llm

OK = {
    "model": "test-model",
    "choices": [{"message": {"content": "Answer [1]."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}
MSGS = [ChatMessage("system", "s"), ChatMessage("user", "u")]


def with_transport(llm: OpenAICompatibleLLM, handler) -> OpenAICompatibleLLM:
    base = llm._client.base_url
    headers = llm._client.headers
    llm._client = httpx.AsyncClient(base_url=base, headers=headers, transport=httpx.MockTransport(handler))
    return llm


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(_):
        return None

    monkeypatch.setattr(openai_compatible.asyncio, "sleep", _sleep)


async def test_groq_defaults_and_request_shape():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.read()
        return httpx.Response(200, json=OK)

    llm = with_transport(GroqLLM("some-groq-model", api_key="gsk_test"), handler)
    resp = await llm.generate(MSGS, max_tokens=50)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    assert b'"model":"some-groq-model"' in seen["body"].replace(b" ", b"")
    assert resp.provider == "groq" and resp.text == "Answer [1]."
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (12, 3)


def test_groq_requires_api_key():
    with pytest.raises(ProviderConfigurationError):
        GroqLLM("m", api_key=None)


async def test_retries_rate_limit_then_succeeds():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "rate limited"})
        return httpx.Response(200, json=OK)

    llm = with_transport(GroqLLM("m", api_key="k"), handler)
    assert (await llm.generate(MSGS, max_tokens=10)).text == "Answer [1]."
    assert len(calls) == 3


async def test_gives_up_after_max_retries():
    llm = with_transport(GroqLLM("m", api_key="k", max_retries=2), lambda r: httpx.Response(503))
    with pytest.raises(ProviderError, match="503"):
        await llm.generate(MSGS, max_tokens=10)


async def test_client_errors_are_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "model not found"}})

    llm = with_transport(GroqLLM("m", api_key="k"), handler)
    with pytest.raises(ProviderError, match="400"):
        await llm.generate(MSGS, max_tokens=10)
    assert len(calls) == 1


def test_registry_builds_groq_and_ignores_local_base_url():
    llm = build_llm(
        LLMSettings(provider="groq", model="m", api_key="gsk_x", base_url="http://localhost:11434/v1")
    )
    assert isinstance(llm, GroqLLM)
    assert str(llm._client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"

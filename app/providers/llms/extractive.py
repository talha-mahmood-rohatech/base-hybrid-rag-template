"""Deterministic, offline "LLM" that answers by quoting the most relevant context sentences.

Useful for CI, air-gapped smoke tests and as a safe fallback. It never invents content:
every sentence it emits is copied verbatim from a numbered context block, followed by that
block's citation marker.
"""

from __future__ import annotations

import re
import time

from app.generation.prompts import NO_ANSWER
from app.providers.llms.base import ChatMessage, LLMProvider, LLMResponse, LLMUsage

_BLOCK_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.S | re.M)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "do",
        "does",
        "did",
        "our",
        "your",
        "my",
        "their",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "can",
        "i",
        "we",
        "you",
    ]
)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1}


class ExtractiveLLM(LLMProvider):
    provider = "extractive"

    def __init__(self, model: str = "extractive-v1", max_sentences: int = 3) -> None:
        super().__init__(model)
        self.max_sentences = max_sentences

    async def generate(
        self, messages: list[ChatMessage], *, max_tokens: int, temperature: float = 0.0
    ) -> LLMResponse:
        start = time.perf_counter()
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context, _, question = user.partition("QUESTION:")
        q_terms = _terms(question)
        scored: list[tuple[float, int, str]] = []
        for m in _BLOCK_RE.finditer(context):
            cid, body = int(m.group(1)), m.group(2)
            for sent in _SENT_RE.split(body.strip()):
                sent = " ".join(sent.split())
                if len(sent) < 20:
                    continue
                overlap = len(q_terms & _terms(sent))
                if overlap:
                    scored.append((overlap / (len(q_terms) or 1), cid, sent))
        scored.sort(key=lambda t: (-t[0], t[1]))
        picked, seen = [], set()
        for _, cid, sent in scored:
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent} [{cid}]")
            if len(picked) >= self.max_sentences:
                break
        text = " ".join(picked) if picked else NO_ANSWER
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            usage=LLMUsage(len(user.split()), len(text.split())),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            finish_reason="stop",
        )

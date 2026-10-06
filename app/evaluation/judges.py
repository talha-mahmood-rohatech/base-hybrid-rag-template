"""Answer-quality judges: faithfulness and answer relevance.

* :class:`HeuristicJudge` - deterministic, offline. Faithfulness = share of answer sentences
  whose content words are (mostly) present in the supplied context; answer relevance =
  cosine similarity between question and answer embeddings.
* :class:`LLMJudge` - asks an LLM to grade (0..1) with a strict JSON response.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.generation.prompts import NO_ANSWER
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.llms.base import ChatMessage, LLMProvider

_CITE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9%.,]+")
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
        "been",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "by",
        "at",
        "as",
        "from",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "can",
        "will",
        "may",
        "must",
        "should",
        "not",
        "no",
        "yes",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
    ]
)


def _content_words(text: str) -> list[str]:
    return [
        w.strip(".,") for w in _WORD.findall(text.lower()) if w.strip(".,") and w.strip(".,") not in _STOP
    ]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Judge(ABC):
    name = "judge"

    @abstractmethod
    async def faithfulness(self, answer: str, contexts: Sequence[str]) -> float: ...

    @abstractmethod
    async def answer_relevance(self, question: str, answer: str) -> float: ...


class HeuristicJudge(Judge):
    name = "heuristic"

    def __init__(self, embedder: EmbeddingProvider, support_threshold: float = 0.7) -> None:
        self.embedder = embedder
        self.support_threshold = support_threshold

    async def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        if not answer or answer.strip() == NO_ANSWER:
            return 1.0  # abstaining is never unfaithful
        ctx_words = set(_content_words(" ".join(contexts)))
        sentences = [s for s in _SENT.split(_CITE.sub("", answer)) if _content_words(s)]
        if not sentences:
            return 0.0
        supported = 0
        for s in sentences:
            words = _content_words(s)
            if sum(1 for w in words if w in ctx_words) / len(words) >= self.support_threshold:
                supported += 1
        return supported / len(sentences)

    async def answer_relevance(self, question: str, answer: str) -> float:
        if not answer or answer.strip() == NO_ANSWER:
            return 0.0
        q = await self.embedder.embed_query(question)
        a = (await self.embedder.embed_documents([_CITE.sub("", answer)]))[0]
        return max(0.0, _cosine(q, a))


class LLMJudge(Judge):
    name = "llm"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def _score(self, instruction: str) -> float:
        resp = await self.llm.generate(
            [
                ChatMessage(
                    "system", 'You are a strict evaluator. Reply with JSON only: {"score": <number 0..1>}'
                ),
                ChatMessage("user", instruction),
            ],
            max_tokens=50,
        )
        m = re.search(r"\{.*\}", resp.text, re.S)
        try:
            return max(0.0, min(1.0, float(json.loads(m.group(0))["score"]))) if m else 0.0
        except (ValueError, KeyError, TypeError):
            return 0.0

    async def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        ctx = "\n\n".join(contexts)
        return await self._score(
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nWhat fraction of the answer's claims is fully supported by "
            "the context? An explicit 'I don't know' counts as 1."
        )

    async def answer_relevance(self, question: str, answer: str) -> float:
        return await self._score(
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nHow directly and completely does the answer address "
            "the question (0 = not at all, 1 = fully)?"
        )

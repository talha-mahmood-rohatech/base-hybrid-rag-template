"""Token counting used for chunk sizing and context budgets.

Uses tiktoken when available; falls back to a deterministic regex approximation so the
platform keeps working fully offline.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

_APPROX_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text, disallowed_special=()))


class ApproximateTokenCounter:
    """~1 token per word/punctuation mark, with long words split every 4 chars."""

    name = "approximate"

    def count(self, text: str) -> int:
        total = 0
        for tok in _APPROX_RE.findall(text):
            total += max(1, (len(tok) + 3) // 4) if len(tok) > 8 else 1
        return total


@lru_cache(maxsize=4)
def get_token_counter(encoding: str = "cl100k_base") -> TokenCounter:
    try:
        return TiktokenCounter(encoding)
    except Exception as exc:  # pragma: no cover - depends on network/cache
        logger.warning("tiktoken unavailable (%s); using approximate token counter", exc)
        return ApproximateTokenCounter()

"""Retrying HTTP calls for model APIs (rate limits, 5xx, flaky networks / TLS interception)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

from app.core.errors import ProviderError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_RETRY_WAIT_S = 30.0


async def request_with_retries(
    send: Callable[[], Awaitable[httpx.Response]], *, what: str, max_retries: int = 3
) -> httpx.Response:
    """Call ``send`` until it returns a non-retryable response; honours ``Retry-After``."""
    for attempt in range(max_retries + 1):
        try:
            resp = await send()
        except httpx.TransportError as exc:
            if attempt == max_retries:
                raise ProviderError(f"{what} failed: {exc!r}") from exc
            wait = min(MAX_RETRY_WAIT_S, 2.0**attempt)
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRYABLE_STATUS or attempt == max_retries:
                raise ProviderError(f"{what} failed ({resp.status_code}): {resp.text[:300]}")
            try:
                wait = float(resp.headers.get("retry-after", 2.0**attempt))
            except ValueError:
                wait = 2.0**attempt
            wait = min(MAX_RETRY_WAIT_S, max(0.5, wait))
        logger.warning("%s retry", what, extra={"attempt": attempt + 1, "wait_s": wait})
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover

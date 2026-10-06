"""API key generation and hashing.

Keys are random, shown once, and only their SHA-256 hash is persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

KEY_PREFIX = "rag_"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

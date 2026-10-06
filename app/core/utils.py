from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

CHUNK_ID_NAMESPACE = uuid.UUID("6f1c3a52-7d0e-4c4b-9a55-1f3f1b0d7e21")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def deterministic_chunk_id(document_version_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    """Stable chunk id: re-ingesting the same version upserts the same Qdrant points."""
    return uuid.uuid5(CHUNK_ID_NAMESPACE, f"{document_version_id}:{chunk_index}")


def slugify(value: str, max_len: int = 64) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:max_len] or "x"


class Stopwatch:
    """Collects named stage timings in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 2)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)

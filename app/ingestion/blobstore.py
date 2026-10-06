"""Raw document storage. Local filesystem (Docker volume) in V1; swap for S3/GCS later."""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class BlobStore(ABC):
    @abstractmethod
    async def put(self, tenant_id: uuid.UUID, checksum: str, data: bytes, suffix: str = "") -> str: ...

    @abstractmethod
    async def get(self, uri: str) -> bytes: ...

    @abstractmethod
    async def delete(self, uri: str) -> None: ...


class LocalBlobStore(BlobStore):
    """Content-addressed per tenant: ``file://<tenant>/<ab>/<checksum><suffix>``."""

    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir).resolve()

    def _path(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise ValueError(f"Unsupported blob uri {uri}")
        path = (self.base / uri.removeprefix("file://")).resolve()
        if self.base not in path.parents:
            raise ValueError("Blob path escapes storage root")
        return path

    async def put(self, tenant_id: uuid.UUID, checksum: str, data: bytes, suffix: str = "") -> str:
        suffix = "".join(ch for ch in suffix.lower() if ch.isalnum() or ch == ".")[:10]
        uri = f"file://{tenant_id}/{checksum[:2]}/{checksum}{suffix}"
        path = self._path(uri)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)

        await asyncio.to_thread(_write)
        return uri

    async def get(self, uri: str) -> bytes:
        return await asyncio.to_thread(self._path(uri).read_bytes)

    async def delete(self, uri: str) -> None:
        await asyncio.to_thread(self._path(uri).unlink, True)

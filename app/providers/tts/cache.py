"""Disk cache for synthesized speech (repeated answers and fixed prompts are free after once)."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from app.providers.tts.base import TextToSpeech


class CachedTTS(TextToSpeech):
    def __init__(self, inner: TextToSpeech, cache_dir: str | Path) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.voice = inner.voice
        self.mime_type = inner.mime_type
        self._dir = Path(cache_dir)

    @property
    def enabled(self) -> bool:
        return self.inner.enabled

    def cache_signature(self) -> str:
        return self.inner.cache_signature()

    def path_for(self, text: str, language: str) -> Path:
        # Anything that changes the audio for the same text (voice, speed, model) is in the key.
        sig = hashlib.sha256(f"{self.inner.cache_signature()}|{language}|{text}".encode()).hexdigest()[:24]
        return self._dir / language / f"{sig}.bin"

    async def synthesize(self, text: str, *, language: str) -> bytes:
        if not text.strip():
            return b""
        path = self.path_for(text, language)
        if await asyncio.to_thread(path.exists):
            return await asyncio.to_thread(path.read_bytes)
        data = await self.inner.synthesize(text, language=language)
        if data:  # never persist an empty result - a transient failure must stay transient

            def _write() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(path)

            await asyncio.to_thread(_write)
        return data

    async def aclose(self) -> None:
        await self.inner.aclose()

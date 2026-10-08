from __future__ import annotations

from abc import ABC, abstractmethod


class TextToSpeech(ABC):
    provider: str = "base"
    model: str = ""
    voice: str = ""
    mime_type: str = "audio/mpeg"

    @property
    def enabled(self) -> bool:
        return True

    @abstractmethod
    async def synthesize(self, text: str, *, language: str) -> bytes:
        """Return encoded audio (``mime_type``) for ``text`` spoken in ``language`` (ISO code)."""

    def cache_signature(self) -> str:
        """Everything besides text/language that changes the audio (used by the disk cache)."""
        return f"{self.provider}|{self.model}|{self.voice}"

    async def aclose(self) -> None:  # noqa: B027
        pass


class NoTextToSpeech(TextToSpeech):
    """Voice answers as text only (``VOICE__TTS_PROVIDER=none``)."""

    provider = "none"

    @property
    def enabled(self) -> bool:
        return False

    async def synthesize(self, text: str, *, language: str) -> bytes:
        return b""

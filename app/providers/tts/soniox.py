"""Soniox text-to-speech (REST, ``tts-rt-v2``). Ported from the Leap voice agent.

POST https://tts-rt.soniox.com/tts, ``Authorization: Bearer <key>``,
body ``{model, language, voice, audio_format, text, speed}`` -> raw audio bytes.
One voice id speaks every supported language; ``language`` selects which.
"""

from __future__ import annotations

import httpx

from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers.http_retry import request_with_retries
from app.providers.tts.base import TextToSpeech

SONIOX_TTS_URL = "https://tts-rt.soniox.com"
_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg"}


class SonioxTTS(TextToSpeech):
    provider = "soniox"

    def __init__(
        self,
        model: str = "tts-rt-v2",
        *,
        api_key: str | None,
        voice: str = "Nina",
        speed: float = 0.95,
        audio_format: str = "mp3",
        base_url: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError(
                "A Soniox API key is required for text-to-speech (VOICE__TTS_API_KEY)"
            )
        if not 0.7 <= speed <= 1.3:
            raise ProviderConfigurationError("Soniox speed must be between 0.7 and 1.3")
        self.model = model
        self.voice = voice
        self.speed = speed
        self.audio_format = audio_format
        self.mime_type = _MIME.get(audio_format, "application/octet-stream")
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=(base_url or SONIOX_TTS_URL).rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    def cache_signature(self) -> str:
        return f"{self.provider}|{self.model}|{self.voice}|{self.speed}|{self.audio_format}"

    async def synthesize(self, text: str, *, language: str) -> bytes:
        if not text.strip():
            return b""
        body = {
            "model": self.model,
            "language": language,
            "voice": self.voice,
            "audio_format": self.audio_format,
            "text": text,
            "speed": self.speed,
        }
        resp = await request_with_retries(
            lambda: self._client.post("/tts", json=body),
            what="Soniox text-to-speech",
            max_retries=self.max_retries,
        )
        if not resp.content:
            # A 200 with an empty body is not usable audio; never let it be cached.
            raise ProviderError(f"Soniox returned empty audio for language {language!r}")
        return resp.content

    async def aclose(self) -> None:
        await self._client.aclose()

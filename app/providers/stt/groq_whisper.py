"""Groq Whisper speech-to-text (OpenAI-compatible ``/audio/transcriptions``).

Ported from the Leap voice agent's ``groq_stt`` adapter. Uses ``verbose_json`` so segment
``avg_logprob`` / ``no_speech_prob`` are available for the reliability gate.
"""

from __future__ import annotations

import statistics
from typing import Any

import httpx

from app.core.errors import ProviderConfigurationError
from app.providers.http_retry import request_with_retries
from app.providers.stt.base import SpeechToText, Transcript

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqWhisperSTT(SpeechToText):
    provider = "groq"

    def __init__(
        self,
        model: str = "whisper-large-v3",
        *,
        api_key: str | None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ProviderConfigurationError(
                "A Groq API key is required for speech-to-text (VOICE__STT_API_KEY)"
            )
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=(base_url or GROQ_BASE_URL).rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "utterance.wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        if not audio:
            return Transcript(text="")
        data: dict[str, str] = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            # Forcing the language improves short, single-word answers, which auto-detect
            # can mistake for another language entirely.
            data["language"] = language
        if prompt:
            # Whisper's initial prompt: spelling/vocabulary hints ("Riba, Ijarah, Murabaha ...").
            data["prompt"] = prompt[:896]
        resp = await request_with_retries(
            lambda: self._client.post("/audio/transcriptions", data=data, files={"file": (filename, audio)}),
            what="Groq speech-to-text",
            max_retries=self.max_retries,
        )
        return self.parse(resp.json())

    @staticmethod
    def parse(body: dict[str, Any]) -> Transcript:
        segments = body.get("segments") or []
        logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
        no_speech = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]
        return Transcript(
            text=(body.get("text") or "").strip(),
            language=body.get("language"),
            avg_logprob=statistics.fmean(logprobs) if logprobs else None,
            no_speech_prob=max(no_speech) if no_speech else None,
            duration_s=body.get("duration"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

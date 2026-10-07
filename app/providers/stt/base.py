from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Phrases Whisper emits for silence, coughs and room noise (they can come back with a low
# no_speech_prob because they are loud), so they are dropped unconditionally.
HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thank you",
        "thanks for watching.",
        "thanks for watching",
        "you.",
        "you",
        "amen.",
        "amén.",
        "mm-hmm.",
        "mm-hmm",
        "bye.",
    }
)


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None  # as reported by the engine ("en" or "english")
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    duration_s: float | None = None

    def is_reliable(self, min_avg_logprob: float, max_no_speech_prob: float) -> bool:
        """False for empty text, known silence hallucinations, or low-confidence decodes."""
        text = self.text.strip()
        if not text or text.lower() in HALLUCINATIONS:
            return False
        if self.avg_logprob is not None and self.avg_logprob < min_avg_logprob:
            return False
        return not (self.no_speech_prob is not None and self.no_speech_prob > max_no_speech_prob)


class SpeechToText(ABC):
    provider: str = "base"
    model: str = ""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "utterance.wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """Transcribe one complete utterance.

        ``language`` (ISO code) forces the decode; ``prompt`` biases recognition toward domain
        vocabulary (product names, jargon) that a general model would otherwise mishear.
        """

    async def aclose(self) -> None:  # noqa: B027
        pass

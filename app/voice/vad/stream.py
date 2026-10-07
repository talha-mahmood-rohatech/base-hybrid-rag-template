"""Turns a stream of PCM chunks into speech_started / speech_ended events.

Knows nothing about WebSockets, STT or RAG: bytes in, :class:`SpeechEvent` out.
Ported from the Leap voice agent's ``server/vad``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from app.voice.vad.audio import pcm16_to_float32, pcm16_to_wav
from app.voice.vad.config import VadConfig
from app.voice.vad.detector import SileroDetector


class SpeechEventKind(StrEnum):
    STARTED = "speech_started"
    ENDED = "speech_ended"


@dataclass(frozen=True)
class SpeechEvent:
    """``wav`` is set on ENDED only: a complete 16 kHz mono WAV of the utterance."""

    kind: SpeechEventKind
    at_ms: int
    wav: bytes | None = None
    duration_ms: int = 0

    @property
    def is_start(self) -> bool:
        return self.kind is SpeechEventKind.STARTED

    @property
    def is_end(self) -> bool:
        return self.kind is SpeechEventKind.ENDED


def _ms_to_frames(ms: int, cfg: VadConfig) -> int:
    return max(1, round(ms * cfg.sample_rate / 1000 / cfg.frame_samples))


class VadStream:
    """One speaker's audio stream. Not thread-safe by design: feed it from one task."""

    def __init__(self, cfg: VadConfig | None = None) -> None:
        self._cfg = cfg or VadConfig()
        self._detector = SileroDetector(self._cfg)
        self._bytes_per_frame = self._cfg.frame_samples * 2
        self._min_speech_frames = _ms_to_frames(self._cfg.min_speech_ms, self._cfg)
        self._min_silence_frames = _ms_to_frames(self._cfg.min_silence_ms, self._cfg)
        self._pad_frames = _ms_to_frames(self._cfg.speech_pad_ms, self._cfg)
        self._max_frames = _ms_to_frames(self._cfg.max_utterance_ms, self._cfg)
        self._frame_ms = self._cfg.frame_samples * 1000 // self._cfg.sample_rate
        # Chunks arrive at whatever size the browser posts; the model needs exact frames.
        self._partial = bytearray()
        # Speech is confirmed only after min_speech_ms (plus any hold), by which point its
        # opening syllable is behind us; the ring lets the turn start from before the trigger.
        self._preroll: deque[bytes] = deque(maxlen=_ms_to_frames(self._cfg.preroll_ms, self._cfg))
        self._hold_frames = 0
        self._reset_turn()
        self._position_ms = 0

    @property
    def config(self) -> VadConfig:
        return self._cfg

    def _reset_turn(self) -> None:
        self._speech: list[bytes] = []
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._trailing: list[bytes] = []

    def hold(self, ms: int) -> None:
        """Suppress *detection* for the next ``ms`` of audio without dropping capture.

        Used to sit out the assistant's own TTS echo tail. Audio still fills the preroll, so
        a word that starts during the hold is captured whole once detection resumes.
        """
        self._hold_frames = _ms_to_frames(ms, self._cfg) if ms > 0 else 0
        if self._in_speech:
            self._reset_turn()

    def reset(self) -> None:
        """Abandon whatever is buffered (stop tapped, barge-in, typed question...)."""
        self._reset_turn()
        self._preroll.clear()
        self._partial.clear()
        self._detector.reset()

    def feed(self, pcm: bytes) -> list[SpeechEvent]:
        """Push raw 16 kHz mono PCM16LE; returns whatever was decided (usually nothing)."""
        self._partial.extend(pcm)
        events: list[SpeechEvent] = []
        while len(self._partial) >= self._bytes_per_frame:
            frame = bytes(self._partial[: self._bytes_per_frame])
            del self._partial[: self._bytes_per_frame]
            event = self._consume_frame(frame)
            if event is not None:
                events.append(event)
        return events

    def _consume_frame(self, frame: bytes) -> SpeechEvent | None:
        prob = self._detector.probability(pcm16_to_float32(frame))
        self._position_ms += self._frame_ms
        if not self._in_speech:
            return self._while_idle(frame, prob)
        return self._while_speaking(frame, prob)

    def _while_idle(self, frame: bytes, prob: float) -> SpeechEvent | None:
        # Before the hold check: the ring keeps filling while detection is suppressed.
        self._preroll.append(frame)
        if self._hold_frames > 0:
            self._hold_frames -= 1
            self._speech_frames = 0
            return None
        if prob < self._cfg.threshold:
            self._speech_frames = 0
            return None
        self._speech_frames += 1
        if self._speech_frames < self._min_speech_frames:
            return None
        self._in_speech = True
        self._speech = list(self._preroll)
        self._preroll.clear()
        self._silence_frames = 0
        self._trailing = []
        return SpeechEvent(kind=SpeechEventKind.STARTED, at_ms=self._position_ms)

    def _while_speaking(self, frame: bytes, prob: float) -> SpeechEvent | None:
        if prob >= self._cfg.neg_threshold:
            # Provisional trailing silence turned out to be a pause inside the sentence.
            self._speech.extend(self._trailing)
            self._trailing = []
            self._speech.append(frame)
            self._silence_frames = 0
        else:
            self._trailing.append(frame)
            self._silence_frames += 1
        if self._silence_frames >= self._min_silence_frames:
            return self._end_turn()
        if len(self._speech) + len(self._trailing) >= self._max_frames:
            return self._end_turn()  # safety bound, not turn-taking policy
        return None

    def _end_turn(self) -> SpeechEvent:
        frames = self._speech + self._trailing[: self._pad_frames]
        pcm = b"".join(frames)
        duration_ms = len(frames) * self._frame_ms
        self._reset_turn()
        self._preroll.clear()
        return SpeechEvent(
            kind=SpeechEventKind.ENDED,
            at_ms=self._position_ms,
            wav=pcm16_to_wav(pcm, self._cfg.sample_rate),
            duration_ms=duration_ms,
        )

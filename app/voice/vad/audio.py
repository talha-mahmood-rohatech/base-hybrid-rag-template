"""PCM plumbing. No VAD logic here - just the format conversions the module needs."""

from __future__ import annotations

import io
import wave

import numpy as np

_INT16_FULL_SCALE = 32768.0


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1), which is what Silero expects.

    An odd trailing byte is half a sample; drop it rather than misalign every sample after it.
    """
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap headerless PCM in a WAV container so an STT API can decode it."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()

"""Every VAD tunable lives here; nothing else in the module holds a magic number.

Configured from the platform settings as ``VOICE__VAD__<FIELD>`` (e.g.
``VOICE__VAD__MIN_SILENCE_MS=1400``). Ported from the Leap voice agent, where these values
were tuned on a live kiosk.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "silero_vad.onnx"


class VadConfig(BaseModel):
    # --- Audio contract -------------------------------------------------------------------
    # Silero v5/v6 is trained on exactly these. The model accepts other frame sizes and
    # returns garbage rather than erroring, so frame size is asserted, not trusted.
    sample_rate: int = 16000
    frame_samples: int = 512  # 32 ms at 16 kHz

    # --- Speech decision ------------------------------------------------------------------
    # Probability above which a frame counts as speech (Silero's default). Raise in loud
    # rooms, lower if quiet talkers get missed.
    threshold: float = Field(default=0.5, ge=0, le=1)
    # Hysteresis: once speaking, only a drop below this starts the silence countdown, so a
    # voice hovering at the boundary does not flicker start/stop.
    neg_threshold: float = Field(default=0.35, ge=0, le=1)

    # --- Turn shaping ---------------------------------------------------------------------
    # Ignore bursts shorter than this (a cough, a door).
    min_speech_ms: int = 250
    # How long the speaker must stay quiet before the turn is over. The first knob to reach
    # for if turns feel too eager (cut off mid-thought) or too sluggish.
    min_silence_ms: int = 1000
    # Audio kept either side of the speech so the first phoneme and last consonant survive.
    speech_pad_ms: int = 200
    # Pre-speech ring. Must exceed echo hold + min_speech_ms + speech_pad_ms, or the front
    # of the first word is lost (Leap: "spin" reached STT as "been").
    preroll_ms: int = 1000

    # --- Safety bound ---------------------------------------------------------------------
    # Memory bound for a client that streams forever (stuck mic, fault, bad actor).
    max_utterance_ms: int = 30_000

    # --- Runtime --------------------------------------------------------------------------
    model_path: Path = DEFAULT_MODEL_PATH
    # One ORT thread per process session: a 2 MB model on 32 ms frames needs no more.
    num_threads: int = 1

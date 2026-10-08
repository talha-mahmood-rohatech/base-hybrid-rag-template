"""Silero VAD, run directly on onnxruntime.

Why not ``pip install silero-vad``: that package hard-depends on torch + torchaudio
(~250 MB) even on its ONNX path, to run the 2 MB model vendored in ``data/``. The inference
contract below is small and stable, so the module owns it instead.

The model is the official ``silero_vad.onnx`` (MIT, see ``data/LICENSE`` and
https://github.com/snakers4/silero-vad).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.voice.vad.config import VadConfig

# Silero v5/v6 conditions each frame on the 64 samples immediately preceding it, passed in
# as part of the input tensor. Omitting them doesn't error - it silently degrades every
# probability - so they are not optional.
_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)

_sessions: dict[str, Any] = {}
_session_lock = threading.Lock()


def _shared_session(cfg: VadConfig) -> Any:
    """One ORT session per model file for the whole process: it holds no per-stream state
    (that lives in SileroDetector) and ``run`` is thread-safe."""
    key = str(cfg.model_path)
    session = _sessions.get(key)
    if session is None:
        with _session_lock:
            session = _sessions.get(key)
            if session is None:
                import onnxruntime

                opts = onnxruntime.SessionOptions()
                opts.inter_op_num_threads = cfg.num_threads
                opts.intra_op_num_threads = cfg.num_threads
                session = onnxruntime.InferenceSession(
                    key, providers=["CPUExecutionProvider"], sess_options=opts
                )
                _sessions[key] = session
    return session


class SileroDetector:
    """Scores one audio stream frame by frame. Stateful: one instance per connection."""

    def __init__(self, cfg: VadConfig) -> None:
        self._cfg = cfg
        self._session = _shared_session(cfg)
        self._sr = np.array(cfg.sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget the stream; carrying LSTM state across turns leaks the previous tail."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for exactly one ``frame_samples`` frame."""
        if frame.shape[-1] != self._cfg.frame_samples:
            raise ValueError(f"expected {self._cfg.frame_samples} samples, got {frame.shape[-1]}")
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT_SAMPLES:]
        return float(out[0][0])

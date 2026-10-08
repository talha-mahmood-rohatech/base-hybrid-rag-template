"""Silero VAD (ported from the Leap voice agent): real model, synthetic speech, no mocks.

The point of replacing the energy detector was that a loud non-voice (aircon,
a tone, a chair) read as speech while a normal talker didn't clear the bar.
test_ignores_loud_non_speech is the regression test for exactly that.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.voice.vad import SpeechEventKind, VadConfig, VadStream

SR = 16000


def _pcm(sig: np.ndarray) -> bytes:
    return (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes()


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SR * ms // 1000, dtype=np.float32)


def _resonate(x: np.ndarray, fc: float, bw: float) -> np.ndarray:
    """Two-pole resonator - one formant."""
    r = np.exp(-np.pi * bw / SR)
    a1, a2 = -2 * r * np.cos(2 * np.pi * fc / SR), r * r
    y = np.zeros_like(x)
    for i in range(2, len(x)):
        y[i] = x[i] - a1 * y[i - 1] - a2 * y[i - 2]
    return y / (np.abs(y).max() + 1e-9)


# Formant triples for a few vowels. Silero is trained on real speech, so a
# *sustained* monotone vowel decays to non-speech after ~500ms - articulation
# has to keep moving. Cycling these gives a syllable train it reads as speech
# for as long as we need, where noise of the same loudness scores ~0.04.
_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240)]


def _syllable(ms: int, vowel: tuple[float, float, float], f0: float) -> np.ndarray:
    t = np.arange(SR * ms // 1000) / SR
    track = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(track) / SR
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 40))
    for fc in vowel:
        sig = _resonate(sig, fc, fc / 10)
    ramp = np.clip(t / 0.02, 0, 1) * np.clip((t[-1] - t) / 0.02, 0, 1)
    return sig * ramp * 0.8


def _voice(ms: int) -> np.ndarray:
    """A synthetic syllable train - the stand-in for a talking visitor."""
    out, i = [], 0
    while sum(len(c) for c in out) < SR * ms // 1000:
        out.append(_syllable(140, _VOWELS[i % len(_VOWELS)], 110 + 15 * (i % 3)))
        out.append(np.zeros(SR * 20 // 1000, dtype=np.float32))  # consonant gap
        i += 1
    return np.concatenate(out)[: SR * ms // 1000].astype(np.float32)


@pytest.fixture
def cfg() -> VadConfig:
    # Shorter than production so a test doesn't have to synthesise seconds of
    # trailing silence; the logic under test is identical.
    return VadConfig(min_silence_ms=300, min_speech_ms=100, speech_pad_ms=100)


def _kinds(events) -> list[SpeechEventKind]:
    return [e.kind for e in events]


def test_ignores_loud_non_speech(cfg):
    """The bug that motivated the rewrite: energetic non-voice must stay silent."""
    rng = np.random.default_rng(0)
    t = np.arange(SR * 2) / SR
    for name, sig in (
        ("silence", _silence(2000)),
        ("white noise", rng.normal(0, 0.3, SR * 2).astype(np.float32)),
        ("300Hz tone", (0.6 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)),
        ("hvac rumble", (0.7 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)),
    ):
        events = VadStream(cfg).feed(_pcm(sig))
        assert events == [], f"{name} was detected as speech"


def test_detects_speech_and_emits_wav(cfg):
    stream = VadStream(cfg)
    events = stream.feed(_pcm(np.concatenate([_silence(300), _voice(800), _silence(800)])))

    assert _kinds(events) == [SpeechEventKind.STARTED, SpeechEventKind.ENDED]
    end = events[1]
    assert end.wav is not None

    with wave.open(io.BytesIO(end.wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        captured_ms = w.getnframes() * 1000 // SR

    # Every millisecond of speech survives - clipping the front is the bug that
    # sent "spin" to STT as "been". The preroll additionally keeps the lead-in
    # ahead of it, but the long trailing silence must still be trimmed off.
    assert captured_ms >= 800, "speech was clipped"
    assert captured_ms <= 300 + 800 + 2 * cfg.speech_pad_ms, "trailing silence not trimmed"


def test_pause_mid_sentence_stays_one_turn(cfg):
    """A visitor pausing mid-thought must not be split into two fragments -
    the failure that had the old detector's silence window pushed to 2s.
    """
    gap = cfg.min_silence_ms // 2
    audio = np.concatenate([_voice(500), _silence(gap), _voice(500), _silence(800)])
    events = VadStream(cfg).feed(_pcm(audio))

    assert _kinds(events) == [SpeechEventKind.STARTED, SpeechEventKind.ENDED]


def test_chunk_size_does_not_change_the_result(cfg):
    """The browser posts whatever the worklet produces; framing is our job."""
    audio = _pcm(np.concatenate([_silence(300), _voice(700), _silence(800)]))

    whole = VadStream(cfg).feed(audio)

    dribbled, stream = [], VadStream(cfg)
    for i in range(0, len(audio), 333):  # deliberately not frame-aligned
        dribbled += stream.feed(audio[i : i + 333])

    assert _kinds(whole) == _kinds(dribbled)
    assert whole[1].wav == dribbled[1].wav


def test_hold_suppresses_detection_but_not_capture(cfg):
    """The assistant's echo tail must not open a turn - but the visitor answering over
    it must still be captured from their first phoneme.
    """
    stream = VadStream(cfg)
    stream.hold(400)

    # 400ms of "echo" while detection is held: nothing may fire.
    assert stream.feed(_pcm(_voice(400))) == []


def test_word_starting_inside_the_hold_is_not_clipped(cfg):
    """The regression that made "spin" arrive at STT as "been": the front of
    the word was being thrown away while the mic sat closed for the guard.
    Holding *detection* instead must keep every sample.
    """
    hold_ms = 300
    speech_ms = 900

    held = VadStream(cfg)
    held.hold(hold_ms)
    # The visitor starts talking immediately - i.e. inside the hold window.
    held_events = held.feed(_pcm(np.concatenate([_voice(speech_ms), _silence(800)])))

    plain = VadStream(cfg)
    plain_events = plain.feed(_pcm(np.concatenate([_voice(speech_ms), _silence(800)])))

    assert SpeechEventKind.ENDED in _kinds(held_events)

    def captured(events):
        end = next(e for e in events if e.kind is SpeechEventKind.ENDED)
        with wave.open(io.BytesIO(end.wav)) as w:
            return w.getnframes() * 1000 // SR

    # Detection was delayed, so the turn is confirmed later - but because the
    # preroll kept filling throughout, essentially the same audio is kept.
    # A closed mic would have lost ~hold_ms off the front instead.
    lost = captured(plain_events) - captured(held_events)
    assert lost < 100, f"{lost}ms of the word was clipped by the hold"


def test_preroll_reaches_back_past_the_hold(cfg):
    """The preroll must outlast hold + min_speech + pad, or the front of the
    word is gone no matter how the hold behaves.
    """

    prod = VadConfig()
    assert prod.preroll_ms > 500 + prod.min_speech_ms + prod.speech_pad_ms, (
        "preroll_ms must exceed ECHO_GUARD_MS + min_speech_ms + speech_pad_ms"
    )


def test_reset_discards_the_turn_in_progress(cfg):
    """Tap-to-stop, barge-in and screen changes all land here: audio captured
    for an abandoned turn must never surface as the next turn's transcript.
    """
    stream = VadStream(cfg)
    assert _kinds(stream.feed(_pcm(_voice(400)))) == [SpeechEventKind.STARTED]

    stream.reset()

    assert stream.feed(_pcm(_silence(800))) == []


def test_max_utterance_bound_releases_the_buffer():
    """Safety bound: a client that streams forever must not grow the buffer
    without limit.
    """
    cfg = VadConfig(min_speech_ms=100, min_silence_ms=300, max_utterance_ms=600)
    events = VadStream(cfg).feed(_pcm(_voice(2000)))

    assert SpeechEventKind.ENDED in _kinds(events)
    assert events[-1].duration_ms <= 900


def test_odd_byte_count_does_not_desync(cfg):
    """Half a sample at a chunk boundary must not misalign everything after."""
    audio = _pcm(np.concatenate([_silence(300), _voice(700), _silence(800)]))
    stream = VadStream(cfg)
    events = stream.feed(audio[:1001]) + stream.feed(audio[1001:])

    assert SpeechEventKind.ENDED in _kinds(events)

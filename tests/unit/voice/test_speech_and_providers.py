"""Speech-text cleanup, Groq Whisper STT, Soniox TTS and the TTS disk cache (no network)."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import ProviderConfigurationError, ProviderError
from app.providers import http_retry
from app.providers.registry import build_stt, build_tts
from app.providers.stt.base import Transcript
from app.providers.stt.groq_whisper import GroqWhisperSTT
from app.providers.tts.base import NoTextToSpeech
from app.providers.tts.cache import CachedTTS
from app.providers.tts.soniox import SonioxTTS
from app.voice.speech import for_speech, to_iso_language


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(_):
        return None

    monkeypatch.setattr(http_retry.asyncio, "sleep", _sleep)


def mock(client_owner, handler):
    old = client_owner._client
    client_owner._client = httpx.AsyncClient(
        base_url=old.base_url, headers=old.headers, transport=httpx.MockTransport(handler)
    )


# --- speech text ----------------------------------------------------------------------------
def test_for_speech_strips_citations_and_markdown():
    answer = (
        "The two kinds of riba are:\n\n"
        "1. **Riba An-Nasiyah** - excess on a loan [1]\n"
        "2. **Riba Al-Fadl** - unequal exchange【2】\n\n"
        "See [the FAQ](https://x.example) for `details` [1, 2]."
    )
    assert for_speech(answer) == (
        "The two kinds of riba are: Riba An-Nasiyah - excess on a loan. "
        "Riba Al-Fadl - unequal exchange. See the FAQ for details."
    )


def test_for_speech_tables_headings_and_truncation():
    text = "## Limits\n| Product | Limit |\n|---|---|\n| Tractor | PKR 2.5m |"
    assert for_speech(text) == "Limits. Product, Limit. Tractor, PKR 2.5m."
    long = " ".join(f"Sentence number {i} is here." for i in range(100))
    short = for_speech(long, max_chars=120)
    assert len(short) <= 120 and short.endswith(".")


def test_to_iso_language():
    assert to_iso_language("english") == "en"
    assert to_iso_language("Urdu") == "ur"
    assert to_iso_language("en") == "en"
    assert to_iso_language("klingon") is None and to_iso_language(None) is None


# --- transcripts ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "t,ok",
    [
        (Transcript("What is Ijarah?", avg_logprob=-0.2, no_speech_prob=0.01), True),
        (Transcript("", avg_logprob=-0.2), False),
        (Transcript("Thank you.", avg_logprob=-0.1, no_speech_prob=0.0), False),  # Whisper hallucination
        (Transcript("mumble", avg_logprob=-2.0), False),
        (Transcript("noise", no_speech_prob=0.95), False),
    ],
)
def test_reliability_gate(t, ok):
    assert t.is_reliable(-1.5, 0.85) is ok


# --- Groq Whisper ---------------------------------------------------------------------------
VERBOSE = {
    "text": " What is the financing limit for a tractor? ",
    "language": "english",
    "duration": 2.4,
    "segments": [
        {"avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"avg_logprob": -0.4, "no_speech_prob": 0.05},
    ],
}


async def test_groq_stt_request_and_parse():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.read()
        return httpx.Response(200, json=VERBOSE)

    stt = GroqWhisperSTT("whisper-large-v3", api_key="gsk_x")
    mock(stt, handler)
    t = await stt.transcribe(b"RIFFfakewav", language="en")
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer gsk_x"
    body = seen["body"]
    for field in (
        b'name="model"',
        b"whisper-large-v3",
        b'name="response_format"',
        b"verbose_json",
        b'name="language"',
        b'filename="utterance.wav"',
        b"RIFFfakewav",
    ):
        assert field in body
    assert t.text == "What is the financing limit for a tractor?"
    assert t.language == "english" and t.duration_s == 2.4
    assert t.avg_logprob == pytest.approx(-0.3) and t.no_speech_prob == pytest.approx(0.05)


async def test_groq_stt_retries_rate_limits_then_fails_cleanly():
    calls = []

    def handler(req):
        calls.append(1)
        return (
            httpx.Response(429, headers={"retry-after": "1"})
            if len(calls) < 3
            else httpx.Response(200, json=VERBOSE)
        )

    stt = GroqWhisperSTT(api_key="k")
    mock(stt, handler)
    assert (await stt.transcribe(b"x")).text.startswith("What is")
    assert len(calls) == 3

    stt2 = GroqWhisperSTT(api_key="k", max_retries=1)
    mock(stt2, lambda r: httpx.Response(400, json={"error": "bad audio"}))
    with pytest.raises(ProviderError, match="400"):
        await stt2.transcribe(b"x")


async def test_empty_audio_skips_the_api():
    stt = GroqWhisperSTT(api_key="k")
    mock(stt, lambda r: pytest.fail("API must not be called"))
    assert (await stt.transcribe(b"")).text == ""


# --- Soniox TTS + cache ---------------------------------------------------------------------
async def test_soniox_request_shape():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["json"] = json.loads(req.read())
        return httpx.Response(200, content=b"ID3mp3bytes", headers={"content-type": "audio/mpeg"})

    tts = SonioxTTS(api_key="s", voice="Nina", speed=0.95)
    mock(tts, handler)
    audio = await tts.synthesize("Hello there.", language="en")
    assert audio == b"ID3mp3bytes"
    assert seen["url"] == "https://tts-rt.soniox.com/tts"
    assert seen["json"] == {
        "model": "tts-rt-v2",
        "language": "en",
        "voice": "Nina",
        "audio_format": "mp3",
        "text": "Hello there.",
        "speed": 0.95,
    }
    assert tts.mime_type == "audio/mpeg"


async def test_soniox_empty_body_is_an_error():
    tts = SonioxTTS(api_key="s")
    mock(tts, lambda r: httpx.Response(200, content=b""))
    with pytest.raises(ProviderError, match="empty audio"):
        await tts.synthesize("Hi.", language="en")


def test_soniox_speed_bounds_and_key():
    with pytest.raises(ProviderConfigurationError):
        SonioxTTS(api_key="s", speed=2.0)
    with pytest.raises(ProviderConfigurationError):
        SonioxTTS(api_key=None)


async def test_cache_hits_disk_and_never_stores_empty(tmp_path):
    calls = []

    def handler(req):
        calls.append(json.loads(req.read())["text"])
        return httpx.Response(200, content=b"audio-" + str(len(calls)).encode())

    inner = SonioxTTS(api_key="s", voice="Nina")
    mock(inner, handler)
    cached = CachedTTS(inner, tmp_path)
    first = await cached.synthesize("Same line.", language="en")
    second = await cached.synthesize("Same line.", language="en")
    assert first == second == b"audio-1" and len(calls) == 1
    await cached.synthesize("Same line.", language="ur")  # language is part of the key
    assert len(calls) == 2

    inner.voice = "Kayla"  # voice changes the audio -> different cache entry
    await cached.synthesize("Same line.", language="en")
    assert len(calls) == 3

    mock(inner, lambda r: httpx.Response(200, content=b""))
    with pytest.raises(ProviderError):
        await cached.synthesize("Never cached.", language="en")
    assert not cached.path_for("Never cached.", "en").exists()


# --- registry -------------------------------------------------------------------------------
def test_registry_reuses_groq_llm_key_for_stt_and_supports_text_only():
    s = Settings(
        _env_file=None,
        llm={"provider": "groq", "model": "m", "api_key": "gsk_shared"},
        voice={"tts_provider": "none"},
    )
    stt = build_stt(s)
    assert isinstance(stt, GroqWhisperSTT)
    assert stt._client.headers["authorization"] == "Bearer gsk_shared"
    assert isinstance(build_tts(s), NoTextToSpeech)

    with pytest.raises(ProviderConfigurationError):
        build_tts(Settings(_env_file=None, voice={"tts_provider": "soniox"}))


def test_split_for_tts_short_first_piece_then_larger():
    from app.voice.speech import split_for_tts

    text = " ".join(f"This is sentence {i} of a long spoken answer." for i in range(12))
    pieces = split_for_tts(text, max_chars=200, first_max_chars=60)
    assert " ".join(pieces) == text
    assert len(pieces[0]) <= 60 and all(len(p) <= 200 for p in pieces)
    assert max(len(p) for p in pieces[1:]) > 60  # later pieces are allowed to grow
    long_clause = "word " * 100
    assert all(len(p) <= 60 for p in split_for_tts(long_clause, 60, 60))


async def test_groq_stt_sends_vocabulary_prompt():
    seen = {}

    def handler(req):
        seen["body"] = req.read()
        return httpx.Response(200, json=VERBOSE)

    stt = GroqWhisperSTT(api_key="k")
    mock(stt, handler)
    await stt.transcribe(b"x", prompt="Riba, Ijarah, Murabaha")
    assert b'name="prompt"' in seen["body"] and b"Riba, Ijarah, Murabaha" in seen["body"]

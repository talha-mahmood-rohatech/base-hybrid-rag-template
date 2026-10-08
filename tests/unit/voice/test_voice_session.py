"""VoiceSession turn-taking with the real Silero VAD and fake STT / RAG / TTS."""

import asyncio
import base64
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import Settings
from app.providers.stt.base import SpeechToText, Transcript
from app.providers.tts.base import NoTextToSpeech, TextToSpeech
from app.voice.session import VoicePipeline, VoiceSession
from tests.unit.voice.test_vad import _pcm, _silence, _voice


class FakeSTT(SpeechToText):
    provider, model = "fake", "fake-whisper"

    def __init__(self, *transcripts: Transcript) -> None:
        self.queue = list(transcripts)
        self.calls: list[tuple[int, str | None]] = []
        self.prompts: list[str | None] = []

    async def transcribe(self, audio, *, filename="utterance.wav", language=None, prompt=None):
        self.calls.append((len(audio), language))
        self.prompts.append(prompt)
        return self.queue.pop(0) if self.queue else Transcript("")


class FakeTTS(TextToSpeech):
    provider, model, voice = "fake", "fake-tts", "Nina"

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    async def synthesize(self, text, *, language):
        self.calls.append((text, language))
        if self.fail:
            raise RuntimeError("tts down")
        return f"MP3<{language}:{text}>".encode()


class FakeOrchestrator:
    def __init__(self, answer="Ijarah is a lease [1].", delay: float = 0.0) -> None:
        self.answer, self.delay = answer, delay
        self.questions: list[str] = []

    async def query(self, *, tenant_id, knowledge_base_id, query, filters, options):
        self.questions.append(query)
        await asyncio.sleep(self.delay)
        cit = SimpleNamespace(
            to_dict=lambda: {"citation_id": 1, "document_name": "faq.docx", "chunk_id": "c1"}
        )
        return SimpleNamespace(
            answer=self.answer, citations=[cit], trace_id=uuid.uuid4(), counts={"dense_candidates": 3}
        )


def make(stt=None, tts=None, orch=None, **voice):
    settings = Settings(_env_file=None, voice={"vad": {"min_silence_ms": 300, "min_speech_ms": 100}, **voice})
    pipeline = VoicePipeline(settings, orch or FakeOrchestrator(), stt or FakeSTT(), tts or FakeTTS())
    sent: list[dict] = []

    async def send(msgs):
        sent.extend(msgs)

    session = VoiceSession(pipeline, send, tenant_id=uuid.uuid4(), knowledge_base_id=uuid.uuid4())
    return session, sent, pipeline


async def settle(session):
    if session._turn is not None:
        await session._turn


def types(sent):
    return [m["type"] for m in sent]


async def test_spoken_question_full_turn():
    stt = FakeSTT(Transcript("What is Ijarah?", language="english", avg_logprob=-0.2, no_speech_prob=0.01))
    tts = FakeTTS()
    orch = FakeOrchestrator()
    session, sent, _ = make(stt, tts, orch)

    await session.on_audio(_pcm(np.concatenate([_silence(300), _voice(800), _silence(800)])))
    await settle(session)

    assert types(sent) == [
        "speech_started",
        "speech_ended",
        "thinking",
        "transcript",
        "answer",
        "say",
        "listen",
    ]
    assert stt.calls[0][0] > 16000  # the VAD handed STT a real WAV of the utterance
    assert orch.questions == ["What is Ijarah?"]
    answer = sent[4]
    assert (
        answer["text"] == "Ijarah is a lease [1]." and answer["citations"][0]["document_name"] == "faq.docx"
    )
    say = sent[5]
    # Citation markers are not read aloud; Whisper's "english" becomes the TTS language "en".
    assert tts.calls == [("Ijarah is a lease.", "en")]
    assert base64.b64decode(say["audio_b64"]) == b"MP3<en:Ijarah is a lease.>"
    assert say["mime"] == "audio/mpeg" and say["interruptible"] is True


async def test_unintelligible_audio_reopens_mic_silently():
    stt = FakeSTT(Transcript("Thank you.", avg_logprob=-0.1, no_speech_prob=0.0))  # Whisper hallucination
    tts, orch = FakeTTS(), FakeOrchestrator()
    session, sent, _ = make(stt, tts, orch)
    await session.on_utterance(b"RIFF....")
    await settle(session)
    assert types(sent) == ["thinking", "transcript", "listen"]
    assert sent[1]["reliable"] is False and sent[2]["reason"] == "unintelligible"
    assert orch.questions == [] and tts.calls == []


async def test_typed_question_skips_stt_and_drops_captured_audio():
    stt = FakeSTT()
    session, sent, _ = make(stt)
    session.vad.feed(_pcm(_voice(400)))  # half a spoken turn in progress
    await session.on_text("What is Musharakah?")
    await settle(session)
    assert stt.calls == []
    assert types(sent) == ["transcript", "thinking", "answer", "say", "listen"]
    assert session.vad.feed(_pcm(_silence(800))) == []  # the half turn was discarded


async def test_barge_in_cancels_an_answer_in_flight():
    orch = FakeOrchestrator(delay=5)
    tts = FakeTTS()
    session, sent, _ = make(orch=orch, tts=tts)
    await session.on_text("slow question")
    await asyncio.sleep(0.05)
    await session.barge_in()
    assert types(sent)[-2:] == ["stop_audio", "listen"]
    assert "answer" not in types(sent) and tts.calls == []
    assert session._turn is None


async def test_new_question_supersedes_the_previous_one():
    orch = FakeOrchestrator(delay=0.3)
    session, sent, _ = make(orch=orch)
    await session.on_text("first")
    await asyncio.sleep(0.05)  # first question is mid-retrieval
    await session.on_text("second")
    await settle(session)
    assert orch.questions == ["first", "second"]
    answers = [m for m in sent if m["type"] == "answer"]
    assert len(answers) == 1  # the superseded turn never answers


async def test_turn_cancelled_before_it_starts_leaves_nothing_behind(recwarn):
    session, sent, _ = make()
    await session.on_text("first")
    await session.on_text("second")  # supersedes before the first task ever ran
    await settle(session)
    assert [m["text"] for m in sent if m["type"] == "transcript"] == ["second"]
    assert not [w for w in recwarn if "never awaited" in str(w.message)]


async def test_tts_failure_still_delivers_the_text_answer():
    session, sent, _ = make(tts=FakeTTS(fail=True))
    await session.on_text("What is Salam?")
    await settle(session)
    assert types(sent) == ["transcript", "thinking", "answer", "error", "say", "listen"]
    assert sent[3]["code"] == "tts_failed" and sent[4]["audio_b64"] == ""


async def test_text_only_mode_and_forced_stt_language():
    stt = FakeSTT(Transcript("What is Riba?", avg_logprob=-0.1))
    session, sent, _ = make(stt=stt, tts=NoTextToSpeech(), stt_language="en")
    await session.on_utterance(b"RIFF....")
    await settle(session)
    assert stt.calls[0][1] == "en"
    say = next(m for m in sent if m["type"] == "say")
    assert say["audio_b64"] == "" and session.client_config()["tts"]["enabled"] is False


async def test_rag_error_is_reported_and_mic_reopens():
    class Broken(FakeOrchestrator):
        async def query(self, **kw):
            raise RuntimeError("qdrant down")

    session, sent, _ = make(orch=Broken())
    await session.on_text("anything")
    await settle(session)
    assert types(sent)[-2:] == ["error", "listen"] and sent[-1]["reason"] == "error"


@pytest.mark.parametrize("hold", [0, 400])
async def test_echo_hold_suppresses_detection(hold):
    session, sent, _ = make()
    session.hold(hold)
    await session.on_audio(_pcm(_voice(300)))
    assert ("speech_started" in types(sent)) is (hold == 0)


async def test_long_answers_are_voiced_in_order_as_pieces_complete():
    sentences = [f"Sentence number {i} explains one more detail of the product." for i in range(8)]

    class SlowFirstTTS(FakeTTS):
        async def synthesize(self, text, *, language):
            # The first piece finishes last; playback order must still follow the text.
            await asyncio.sleep(0.2 if text.startswith("Sentence number 0") else 0.01)
            return await super().synthesize(text, language=language)

    tts = SlowFirstTTS()
    session, sent, _ = make(
        tts=tts,
        orch=FakeOrchestrator(answer=" ".join(sentences)),
        tts_chunk_chars=120,
        tts_first_chunk_chars=70,
    )
    await session.on_text("tell me everything")
    await settle(session)
    says = [m for m in sent if m["type"] == "say"]
    assert len(says) > 1 and types(sent)[-1] == "listen"
    assert " ".join(m["text"] for m in says) == " ".join(sentences)
    assert len(says[0]["text"]) <= 70 and all(len(m["text"]) <= 120 for m in says)
    assert [base64.b64decode(m["audio_b64"]).decode() for m in says] == [f"MP3<en:{m['text']}>" for m in says]


async def test_vocabulary_prompt_default_and_per_session_override():
    stt = FakeSTT(*[Transcript("What is Riba?", avg_logprob=-0.1)] * 2)
    default_session, _sent, pipeline = make(stt=stt, stt_prompt="Riba, Ijarah")
    await default_session.on_utterance(b"RIFF")
    await settle(default_session)
    custom = VoiceSession(
        pipeline,
        default_session.send,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
        stt_prompt="Murabaha, Musharakah",
    )
    await custom.on_utterance(b"RIFF")
    await settle(custom)
    assert stt.prompts == ["Riba, Ijarah", "Murabaha, Musharakah"]

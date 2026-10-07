"""Voice over the real app: WebSocket + REST, real PostgreSQL/Qdrant/RAG, fake STT/TTS."""

import base64
import io
import time
import uuid
import wave

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient

import app.container as container_module
from app.core.config import WorkerSettings
from app.core.errors import ProviderConfigurationError
from app.main import create_app
from app.providers.stt.base import Transcript
from tests.conftest import TEST_QDRANT_URL, make_settings
from tests.unit.voice.test_vad import _pcm, _silence, _voice
from tests.unit.voice.test_voice_session import FakeSTT, FakeTTS

pytestmark = pytest.mark.integration

DOC = (
    "# Islamic Leasing\n\n## Ijarah\n\nIjarah means transferring the usufruct of an asset to another "
    "person for an agreed period at an agreed consideration. The bank remains the owner of the asset."
)


def _cleanup_collections(prefix: str) -> None:
    cols = httpx.get(f"{TEST_QDRANT_URL}/collections").json()["result"]["collections"]
    for c in cols:
        if c["name"].startswith(prefix):
            httpx.delete(f"{TEST_QDRANT_URL}/collections/{c['name']}")


def _wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def voice_app(migrated_db, tmp_path, monkeypatch):
    question = Transcript("What does Ijarah mean?", language="english", avg_logprob=-0.2, no_speech_prob=0.01)
    stt = FakeSTT(*[question] * 10)
    tts = FakeTTS()
    monkeypatch.setattr(container_module, "build_stt", lambda s: stt)
    monkeypatch.setattr(container_module, "build_tts", lambda s: tts)
    settings = make_settings(tmp_path, worker=WorkerSettings(embedded=True, poll_interval_s=0.2))
    with TestClient(create_app(settings)) as client:
        admin = {"X-Admin-Key": settings.security.admin_api_key.get_secret_value()}
        key = client.post(
            "/v1/admin/tenants", headers=admin, json={"name": "v", "slug": f"voice-{time.time_ns()}"}
        ).json()["api_key"]
        h = {"X-API-Key": key}
        kb = client.post("/v1/knowledge-bases", headers=h, json={"name": "leasing"}).json()["id"]
        job = client.post(
            "/v1/documents", headers=h, data={"knowledge_base_id": kb, "text": DOC, "filename": "ijarah.md"}
        ).json()["job"]["id"]
        for _ in range(100):
            if client.get(f"/v1/ingestion/jobs/{job}", headers=h).json()["status"] == "succeeded":
                break
            time.sleep(0.2)
        else:
            pytest.fail("ingestion did not finish")
        yield client, key, kb, stt, tts
    _cleanup_collections(settings.qdrant.collection_prefix)


def collapse(kinds: list[str]) -> list[str]:
    """Merge consecutive duplicates ('say', 'say', 'say' -> 'say')."""
    return [k for i, k in enumerate(kinds) if i == 0 or kinds[i - 1] != k]


def recv_until(ws, last_type: str, limit: int = 30) -> list[dict]:
    out = []
    for _ in range(limit):
        m = ws.receive_json()
        out.append(m)
        if m["type"] == last_type:
            return out
    raise AssertionError(f"never received {last_type}: {[m['type'] for m in out]}")


def test_voice_websocket_typed_and_spoken_turns(voice_app):
    client, key, kb, stt, _tts = voice_app
    with client.websocket_connect("/v1/voice/ws") as ws:
        ws.send_json({"type": "start", "api_key": key, "knowledge_base_id": kb})
        ready, listen = ws.receive_json(), ws.receive_json()
        assert ready["type"] == "ready" and listen["type"] == "listen"
        assert ready["config"]["sample_rate"] == 16000 and ready["config"]["knowledge_base"] == "leasing"

        # Typed question: transcript -> thinking -> answer (cited) -> say -> listen
        ws.send_json({"type": "text", "text": "What is Ijarah?"})
        msgs = recv_until(ws, "listen")
        assert collapse([m["type"] for m in msgs]) == ["transcript", "thinking", "answer", "say", "listen"]
        answer = msgs[2]
        assert "usufruct" in answer["text"] and answer["citations"][0]["document_name"] == "ijarah.md"
        says = [m for m in msgs if m["type"] == "say"]  # long answers are voiced in pieces
        assert all(base64.b64decode(m["audio_b64"]).startswith(b"MP3<en:") for m in says)
        spoken = " ".join(m["text"] for m in says)
        assert "usufruct" in spoken and "[1]" not in spoken  # citations are not read aloud

        # Spoken question: streamed PCM in odd-sized chunks through the real Silero VAD
        pcm = _pcm(np.concatenate([_silence(300), _voice(900), _silence(1500)]))  # > default min_silence_ms
        for i in range(0, len(pcm), 4096 + 17):
            ws.send_bytes(pcm[i : i + 4096 + 17])
        msgs = recv_until(ws, "listen")
        kinds = collapse([m["type"] for m in msgs])
        assert kinds == [
            "speech_started",
            "speech_ended",
            "thinking",
            "transcript",
            "answer",
            "say",
            "listen",
        ]
        assert msgs[3]["text"] == "What does Ijarah mean?" and msgs[3]["reliable"] is True
        assert "usufruct" in msgs[4]["text"]
        assert stt.calls and stt.calls[-1][0] > 16000 * 2 * 0.9  # ~0.9 s of 16 kHz PCM16 + header

        ws.send_json({"type": "barge_in"})
        assert [m["type"] for m in recv_until(ws, "listen")] == ["stop_audio", "listen"]


def test_user_page_and_developer_console_are_served(voice_app):
    client, *_ = voice_app
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/voice/"
    user = client.get("/voice/")
    assert user.status_code == 200 and 'id="mic"' in user.text
    assert "trace" not in user.text.lower() and "api key" not in user.text.lower()  # no developer details
    console = client.get("/voice/console.html")
    assert console.status_code == 200 and "Voice RAG Console" in console.text
    for asset in (
        "app.js",
        "style.css",
        "console.js",
        "console.css",
        "audio/mic-stream.js",
        "audio/pcm-worklet.js",
    ):
        assert client.get(f"/voice/{asset}").status_code == 200, asset


def test_public_assistant_needs_no_key_only_when_configured(voice_app):
    client, _key, kb, *_ = voice_app
    voice_settings = client.app.state.container.settings.voice
    assert client.get("/v1/voice/public").json()["enabled"] is False
    with client.websocket_connect("/v1/voice/ws") as ws:
        ws.send_json({"type": "start"})  # no key, public mode off -> refused
        assert ws.receive_json()["code"] == "bad_request"

    voice_settings.public_knowledge_base_id = uuid.UUID(kb)
    voice_settings.public_title = "Leasing helper"
    try:
        cfg = client.get("/v1/voice/public").json()
        assert cfg == {"enabled": True, "title": "Leasing helper", "speech": True}
        with client.websocket_connect("/v1/voice/ws") as ws:
            # client-supplied knowledge base / filters are ignored in public mode
            ws.send_json(
                {"type": "start", "knowledge_base_id": str(uuid.uuid4()), "filters": {"document_type": "pdf"}}
            )
            assert ws.receive_json()["type"] == "ready"
            ws.receive_json()  # listen
            ws.send_json({"type": "text", "text": "What is Ijarah?"})
            answer = next(m for m in recv_until(ws, "listen") if m["type"] == "answer")
            assert "usufruct" in answer["text"]
    finally:
        voice_settings.public_knowledge_base_id = None


def test_voice_websocket_rejects_bad_auth(voice_app):
    client, _key, kb, *_ = voice_app
    with client.websocket_connect("/v1/voice/ws") as ws:
        ws.send_json({"type": "start", "api_key": "rag_wrong", "knowledge_base_id": kb})
        assert ws.receive_json()["code"] == "unauthorized"
    with client.websocket_connect("/v1/voice/ws") as ws:
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["code"] == "bad_request"


def test_voice_rest_endpoints(voice_app):
    client, key, kb, *_ = voice_app
    h = {"X-API-Key": key}
    wav = _wav(_pcm(_voice(600)))

    r = client.post("/v1/voice/transcribe", headers=h, files={"file": ("q.wav", wav, "audio/wav")})
    assert r.status_code == 200 and r.json()["text"] == "What does Ijarah mean?" and r.json()["reliable"]

    r = client.post("/v1/voice/speak", headers=h, json={"text": "Ijarah is a lease [1]."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.content == b"MP3<en:Ijarah is a lease.>"

    r = client.post(
        "/v1/voice/ask",
        headers=h,
        data={"knowledge_base_id": kb},
        files={"file": ("q.wav", wav, "audio/wav")},
    )
    body = r.json()
    assert r.status_code == 200 and "usufruct" in body["answer"]
    assert body["citations"] and base64.b64decode(body["audio_b64"]).startswith(b"MP3<en:")
    assert set(body["timings_ms"]) == {"stt_ms", "rag_ms", "tts_ms"}

    assert client.post("/v1/voice/transcribe", files={"file": ("q.wav", wav)}).status_code == 401


def test_voice_unavailable_does_not_break_the_platform(migrated_db, tmp_path, monkeypatch):
    def missing(_):
        raise ProviderConfigurationError("A Soniox API key is required for text-to-speech")

    monkeypatch.setattr(container_module, "build_tts", missing)
    monkeypatch.setattr(container_module, "build_stt", lambda s: FakeSTT())
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        admin = {"X-Admin-Key": settings.security.admin_api_key.get_secret_value()}
        key = client.post(
            "/v1/admin/tenants", headers=admin, json={"name": "n", "slug": f"novoice-{time.time_ns()}"}
        ).json()["api_key"]
        r = client.post("/v1/voice/speak", headers={"X-API-Key": key}, json={"text": "hi"})
        assert r.status_code == 503 and "Soniox" in r.json()["error"]["message"]
        with client.websocket_connect("/v1/voice/ws") as ws:
            assert ws.receive_json()["code"] == "voice_unavailable"
    _cleanup_collections(settings.qdrant.collection_prefix)

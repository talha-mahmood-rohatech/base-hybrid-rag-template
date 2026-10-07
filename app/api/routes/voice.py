"""Voice endpoints: streaming conversation (WebSocket) + one-shot REST helpers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.deps import TenantContext, authenticate_api_key, get_container, get_tenant
from app.container import Container
from app.core.errors import NotFoundError, RAGError, ValidationError
from app.ingestion.service import get_kb
from app.models import KnowledgeBase
from app.voice import messages as M
from app.voice.session import VoicePipeline, VoiceSession
from app.voice.speech import for_speech

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/voice", tags=["voice"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq's Whisper upload limit
AUTH_TIMEOUT_S = 15.0


class VoiceUnavailableError(RAGError):
    code = "voice_unavailable"
    status_code = 503


def pipeline_of(container: Container) -> VoicePipeline:
    if container.stt is None or container.tts is None:
        raise VoiceUnavailableError(container.voice_unavailable_reason or "Voice pipeline is not configured")
    return VoicePipeline(container.settings, container.orchestrator, container.stt, container.tts)


async def _read_audio(file: UploadFile) -> bytes:
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise ValidationError("Audio file is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValidationError("Audio exceeds 25 MB")
    return data


# --- REST ---------------------------------------------------------------------------------
@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="wav/mp3/m4a/webm/ogg/flac"),
    prompt: str | None = Form(default=None, max_length=896, description="Vocabulary hint for STT"),
    _: TenantContext = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    pipeline = pipeline_of(container)
    t0 = time.perf_counter()
    t, reliable = await pipeline.transcribe(await _read_audio(file), file.filename or "audio.wav", prompt)
    return {
        "text": t.text,
        "language": t.language,
        "reliable": reliable,
        "avg_logprob": t.avg_logprob,
        "no_speech_prob": t.no_speech_prob,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    clean: bool = Field(default=True, description="Strip citation markers and markdown before speaking")


@router.post("/speak", response_class=Response)
async def speak(
    body: SpeakRequest,
    _: TenantContext = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> Response:
    pipeline = pipeline_of(container)
    if not pipeline.tts.enabled:
        raise VoiceUnavailableError("Text-to-speech is disabled (VOICE__TTS_PROVIDER=none)")
    text = for_speech(body.text, 5000) if body.clean else body.text
    audio = await pipeline.speak(text, body.language or pipeline.voice.tts_language)
    return Response(content=audio, media_type=pipeline.tts.mime_type)


@router.post("/ask")
async def ask(
    file: UploadFile = File(..., description="One spoken question"),
    knowledge_base_id: uuid.UUID = Form(...),
    top_k: int | None = Form(default=None, ge=1, le=50),
    prompt: str | None = Form(default=None, max_length=896, description="Vocabulary hint for STT"),
    tenant: TenantContext = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    """One-shot voice question: audio in -> transcript, cited answer and spoken answer out."""
    pipeline = pipeline_of(container)
    audio_in = await _read_audio(file)
    async with container.session_factory() as session:
        await get_kb(session, tenant.tenant_id, knowledge_base_id)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    transcript, reliable = await pipeline.transcribe(audio_in, file.filename or "audio.wav", prompt)
    timings["stt_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out: dict[str, Any] = {
        "transcript": {"text": transcript.text, "language": transcript.language, "reliable": reliable},
        "answer": None,
        "citations": [],
        "trace_id": None,
        "audio_b64": None,
        "mime": pipeline.tts.mime_type,
        "timings_ms": timings,
    }
    if not reliable:
        return out
    t1 = time.perf_counter()
    result = await pipeline.ask(
        tenant_id=tenant.tenant_id, kb_id=knowledge_base_id, question=transcript.text, filters={}, top_k=top_k
    )
    timings["rag_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    language = pipeline.speech_language(transcript)
    t2 = time.perf_counter()
    audio_out = await pipeline.speak(
        for_speech(result.answer or "", pipeline.voice.max_spoken_chars), language
    )
    timings["tts_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    out.update(
        answer=result.answer,
        citations=[c.to_dict() for c in result.citations],
        trace_id=str(result.trace_id),
        audio_b64=base64.b64encode(audio_out).decode("ascii") if audio_out else None,
        language=language,
    )
    return out


@router.get("/public")
async def public_config(container: Container = Depends(get_container)) -> dict[str, Any]:
    """What the simple user page needs to know (no secrets): is a key-less assistant enabled?"""
    v = container.settings.voice
    enabled = (
        v.public_knowledge_base_id is not None and container.stt is not None and container.tts is not None
    )
    return {
        "enabled": enabled,
        "title": v.public_title,
        "speech": bool(container.tts and container.tts.enabled),
    }


# --- WebSocket ----------------------------------------------------------------------------
@router.websocket("/ws")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    container: Container = websocket.app.state.container
    send_lock = asyncio.Lock()

    async def send(msgs: list[dict[str, Any]]) -> None:
        async with send_lock:
            for m in msgs:
                await websocket.send_text(json.dumps(m, ensure_ascii=False))

    async def refuse(code: str, message: str, close_code: int) -> None:
        await send([M.error(code, message)])
        await websocket.close(code=close_code)

    try:
        pipeline = pipeline_of(container)
    except VoiceUnavailableError as exc:
        await refuse(exc.code, exc.message, 1011)
        return

    # Browsers cannot set headers on a WebSocket, so the first message authenticates:
    # {"type":"start","api_key":..,"knowledge_base_id":..} - or {"type":"start"} alone for the
    # public assistant when VOICE__PUBLIC_KNOWLEDGE_BASE_ID is configured.
    public_kb_id = container.settings.voice.public_knowledge_base_id
    try:
        first = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_S)
        start = json.loads(first)
        if not isinstance(start, dict) or start.get("type") != "start":
            raise ValueError
        public = not start.get("api_key")
        if public and public_kb_id is None:
            raise ValueError
        kb_id: uuid.UUID = (
            public_kb_id if public and public_kb_id else uuid.UUID(str(start["knowledge_base_id"]))
        )
    except (TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        await refuse(
            "bad_request", 'First message must be {"type":"start","api_key":..,"knowledge_base_id":..}', 1008
        )
        return
    except WebSocketDisconnect:
        return
    try:
        async with container.session_factory() as db:
            if public:
                kb = await db.get(KnowledgeBase, kb_id)
                if kb is None:
                    raise NotFoundError("The public knowledge base is not available")
                tenant_id = kb.tenant_id
            else:
                tenant = await authenticate_api_key(db, str(start["api_key"]))
                kb = await get_kb(db, tenant.tenant_id, kb_id)
                tenant_id = tenant.tenant_id
    except RAGError as exc:
        await refuse(exc.code, exc.message, 1008)
        return

    # Public sessions run on server settings only; authenticated clients may tune retrieval.
    filters = start.get("filters") if not public and isinstance(start.get("filters"), dict) else {}
    top_k = start.get("top_k") if not public and isinstance(start.get("top_k"), int) else None
    stt_prompt = (
        str(start["stt_prompt"])[:896] if not public and isinstance(start.get("stt_prompt"), str) else None
    )
    session = VoiceSession(
        pipeline,
        send,
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        filters=filters,
        top_k=top_k,
        stt_prompt=stt_prompt,
    )
    logger.info("voice session started", extra={"session": session.id, "kb": str(kb.id)})
    await send([M.ready(session.id, {**session.client_config(), "knowledge_base": kb.name}), M.listen()])

    try:
        while True:
            packet = await websocket.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            if packet.get("bytes") is not None:
                await session.on_audio(packet["bytes"])
                continue
            if packet.get("text") is None:
                continue
            try:
                msg = json.loads(packet["text"])
                kind = msg.get("type")
            except (json.JSONDecodeError, AttributeError):
                await send([M.error("bad_request", "Messages must be JSON objects")])
                continue
            if kind == "text":
                text = str(msg.get("text", "")).strip()
                if text:
                    await session.on_text(text[:4000])
            elif kind == "audio":
                try:
                    audio = base64.b64decode(msg.get("data", ""), validate=True)
                except (binascii.Error, ValueError):
                    await send([M.error("bad_request", "audio.data must be base64")])
                    continue
                await session.on_utterance(audio, str(msg.get("filename") or "utterance.wav"))
            elif kind == "barge_in":
                await session.barge_in()
            elif kind == "vad_hold":
                session.hold(int(msg.get("ms", 0)))
            elif kind == "vad_reset":
                session.drop_audio()
            elif kind == "reset":
                await session.reset()
            else:
                await send([M.error("bad_request", f"Unknown message type {kind!r}")])
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await session.close()
        logger.info("voice session closed", extra={"session": session.id})

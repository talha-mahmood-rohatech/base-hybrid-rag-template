"""Messages the voice WebSocket sends to clients - the only contract with the browser.

Client -> server (JSON text unless noted):
  {"type":"start","api_key":"...","knowledge_base_id":"...",       first message (auth); optional
   "filters":{},"top_k":8,"stt_prompt":"Riba, Ijarah, ..."}        filters, top_k, vocabulary hint
  <binary frame>                    raw 16 kHz mono PCM16LE, streamed while the mic is open
  {"type":"text","text":"..."}      typed question (bypasses VAD + STT)
  {"type":"audio","data":"<b64>"}   one complete utterance (any format Whisper accepts; bypasses VAD)
  {"type":"barge_in"}               user interrupted playback: cancel the turn, drop captured audio
  {"type":"vad_hold","ms":500}      keep capturing but suppress detection (assistant's echo tail)
  {"type":"vad_reset"}              drop the utterance in progress (mic tapped off)
  {"type":"reset"}                  cancel everything

Server -> client: one JSON object per frame, built by the functions below.
"""

from __future__ import annotations

from typing import Any


def _msg(type_: str, **data: Any) -> dict[str, Any]:
    return {"type": type_, **data}


def ready(session_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return _msg("ready", session_id=session_id, config=config)


def speech_started() -> dict[str, Any]:
    return _msg("speech_started")


def speech_ended(duration_ms: int) -> dict[str, Any]:
    return _msg("speech_ended", duration_ms=duration_ms)


def transcript(
    text: str, *, language: str | None, reliable: bool, latency_ms: float | None = None
) -> dict[str, Any]:
    return _msg("transcript", text=text, language=language, reliable=reliable, latency_ms=latency_ms)


def thinking() -> dict[str, Any]:
    return _msg("thinking")


def answer(
    text: str, citations: list[dict], trace_id: str, retrieval: dict, latency_ms: float
) -> dict[str, Any]:
    return _msg(
        "answer",
        text=text,
        citations=citations,
        trace_id=trace_id,
        retrieval=retrieval,
        latency_ms=latency_ms,
    )


def say(
    text: str,
    audio_b64: str,
    mime: str,
    *,
    language: str,
    interruptible: bool = True,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    return _msg(
        "say",
        text=text,
        audio_b64=audio_b64,
        mime=mime,
        language=language,
        interruptible=interruptible,
        latency_ms=latency_ms,
    )


def listen(reason: str = "") -> dict[str, Any]:
    return _msg("listen", reason=reason)


def stop_audio() -> dict[str, Any]:
    return _msg("stop_audio")


def error(code: str, message: str) -> dict[str, Any]:
    return _msg("error", code=code, message=message)

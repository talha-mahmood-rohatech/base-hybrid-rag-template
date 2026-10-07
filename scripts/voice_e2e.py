"""Live end-to-end check of the voice pipeline against a running stack - no microphone needed.

For each question: Soniox speaks it as 16 kHz WAV -> the PCM is streamed over /v1/voice/ws in
128 ms chunks (like the browser worklet) -> server Silero VAD -> Groq Whisper -> hybrid RAG ->
Soniox answer audio. The spoken answer is transcribed back with Whisper as a round-trip check.

    python scripts/voice_e2e.py --tenant-slug ztbl --kb islamic-banking --api http://localhost:8000 \
        "What is the financing limit for a rice transplanter?" "Who are the members of the Shariah Board?"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNK_BYTES = 512 * 4 * 2  # 4 Silero frames of 16-bit audio = 128 ms, as the browser worklet posts


async def speak_wav16k(text: str, key: str, voice: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://tts-rt.soniox.com/tts",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "tts-rt-v2",
                "language": "en",
                "voice": voice,
                "text": text,
                "audio_format": "wav",
                "sample_rate": 16000,
                "speed": 1.0,
            },
        )
        r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return w.readframes(w.getnframes())


async def run_question(
    api: str,
    api_key: str,
    kb_id: str,
    question: str,
    pcm: bytes,
    realtime: bool,
    stt_prompt: str | None,
    stop_after_transcript: bool = False,
) -> dict:
    ws_url = api.replace("http", "ws", 1) + "/v1/voice/ws"
    timeline: list[tuple[float, dict]] = []
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        start = {"type": "start", "api_key": api_key, "knowledge_base_id": kb_id}
        if stt_prompt:
            start["stt_prompt"] = stt_prompt
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise SystemExit(f"voice session refused: {ready}")
        await ws.recv()  # listen
        # leading silence, the question, then trailing silence so the VAD ends the turn
        audio = bytes(16000 * 2 // 2) + pcm + bytes(16000 * 2 * 2)
        t0 = time.perf_counter()

        async def stream() -> None:
            for i in range(0, len(audio), CHUNK_BYTES):
                await ws.send(audio[i : i + CHUNK_BYTES])
                if realtime:
                    await asyncio.sleep(0.128)

        sender = asyncio.create_task(stream())
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            timeline.append((time.perf_counter() - t0, msg))
            if stop_after_transcript and msg["type"] == "transcript":
                await ws.send(json.dumps({"type": "reset"}))  # cancel the RAG turn
                break
            if msg["type"] in ("listen", "error") and any(m["type"] == "say" for _, m in timeline):
                break
            if msg["type"] == "listen" and msg.get("reason") in ("unintelligible", "error"):
                break
        await sender
    return {"question": question, "audio_s": len(pcm) / 32000, "timeline": timeline}


async def main() -> None:
    from app.core.config import Settings
    from app.providers.registry import build_stt

    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--tenant-slug", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--realtime", action="store_true", help="pace audio at real time like a live mic")
    ap.add_argument("--stt-prompt", help="vocabulary hint sent in the start message")
    ap.add_argument("--transcripts-only", action="store_true", help="only report what STT heard")
    args = ap.parse_args()

    settings = Settings()
    soniox_key = settings.voice.tts_api_key.get_secret_value() if settings.voice.tts_api_key else ""
    if not soniox_key:
        raise SystemExit("VOICE__TTS_API_KEY is needed to synthesize the test questions")
    api_key = json.loads((ROOT / ".secrets" / f"{args.tenant_slug}.json").read_text())["api_key"]
    async with httpx.AsyncClient() as c:
        kbs = (await c.get(f"{args.api}/v1/knowledge-bases", headers={"X-API-Key": api_key})).json()
    kb = next(k for k in kbs if k["name"] == args.kb)
    stt = build_stt(settings)

    for q in args.questions:
        pcm = await speak_wav16k(q, soniox_key, settings.voice.tts_voice)
        res = await run_question(
            args.api, api_key, kb["id"], q, pcm, args.realtime, args.stt_prompt, args.transcripts_only
        )
        if args.transcripts_only:
            heard = next((m["text"] for _, m in res["timeline"] if m["type"] == "transcript"), None)
            print(
                f"{'OK ' if heard and heard.lower().rstrip('?.') == q.lower().rstrip('?.') else 'DIFF'} said={q!r} heard={heard!r}"
            )
            continue
        print(f"\nQ (spoken, {res['audio_s']:.1f}s audio): {q}")
        for t, m in res["timeline"]:
            detail = ""
            if m["type"] == "transcript":
                detail = f"{m['text']!r} reliable={m['reliable']} stt={m.get('latency_ms')}ms"
            elif m["type"] == "speech_ended":
                detail = f"utterance {m['duration_ms']} ms"
            elif m["type"] == "answer":
                cites = ", ".join(
                    " > ".join((c.get("section") or "").split(" > ")[-2:]) for c in m["citations"]
                )
                detail = f"rag={m['latency_ms']:.0f}ms  {m['text'][:110]!r}  cites=[{cites}]"
            elif m["type"] == "say":
                detail = f"tts={m.get('latency_ms')}ms audio={len(base64.b64decode(m['audio_b64'])) if m['audio_b64'] else 0}B"
            elif m["type"] == "error":
                detail = m["message"]
            print(f"  +{t:6.2f}s  {m['type']:<15} {detail}")
        spoken = b"".join(
            base64.b64decode(m["audio_b64"])
            for _, m in res["timeline"]
            if m["type"] == "say" and m["audio_b64"]
        )
        if spoken:  # MP3 frames concatenate cleanly, so the pieces form one playable answer
            back = await stt.transcribe(spoken, filename="answer.mp3")
            print(f"  round-trip (Whisper on the spoken answer): {back.text[:160]!r}")
    await stt.aclose()


if __name__ == "__main__":
    asyncio.run(main())

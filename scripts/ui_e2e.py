"""Browser test of the user page (/voice/) and the developer console (/voice/console.html) - a real Edge/Chromium with a fake microphone.

The question is spoken by Soniox TTS into a WAV that the browser plays as its microphone, so
the whole path runs exactly as for a user: getUserMedia -> AudioWorklet -> WebSocket PCM ->
Silero VAD -> Whisper -> hybrid RAG -> Soniox -> playback, plus the trace inspector and a
typed follow-up. Screenshots are written to reports/ui/.

    pip install playwright          # dev tool; uses the installed Edge (no browser download)
    python scripts/ui_e2e.py --tenant-slug ztbl --kb islamic-banking --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def question_wav(text: str, key: str, voice: str, lead_s: float) -> Path:
    """Soniox-spoken question with leading silence (the mic is tapped a moment after it opens)."""
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
        pcm = w.readframes(w.getnframes())
    lead = bytes(int(16000 * lead_s) * 2)
    tail = bytes(16000 * 3 * 2)
    out = Path(tempfile.gettempdir()) / "voice_rag_question.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(lead + pcm + tail)
    return out


async def main() -> None:
    from playwright.async_api import async_playwright

    from app.core.config import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-slug", required=True, help="for the developer console")
    ap.add_argument("--kb", required=True, help="for the developer console")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--question", default="What is the financing limit for a rice transplanter?")
    ap.add_argument("--typed", default="How are losses shared in Musharakah?")
    ap.add_argument("--vocab", default="Riba, Ijarah, Murabaha, Musharakah, Mudarabah, Qarz-e-Hasna")
    ap.add_argument("--channel", default="msedge", help="msedge | chrome | chromium")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    settings = Settings()
    soniox = settings.voice.tts_api_key.get_secret_value() if settings.voice.tts_api_key else ""
    if not soniox:
        raise SystemExit("VOICE__TTS_API_KEY is needed to speak the test question")
    api_key = json.loads((ROOT / ".secrets" / f"{args.tenant_slug}.json").read_text())["api_key"]
    wav = await question_wav(args.question, soniox, settings.voice.tts_voice, lead_s=4.0)
    shots = ROOT / "reports" / "ui"
    shots.mkdir(parents=True, exist_ok=True)

    flags = [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav}%noloop",
        "--autoplay-policy=no-user-gesture-required",
    ]
    errors: list[str] = []
    report: dict = {}

    async def new_page(p, width: int, height: int):
        # A fresh browser per page, so the fake microphone starts its recording from the top.
        browser = await p.chromium.launch(
            channel=None if args.channel == "chromium" else args.channel, headless=not args.headed, args=flags
        )
        page = await browser.new_page(viewport={"width": width, "height": height})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        return browser, page

    async with async_playwright() as p:
        # =========================== user page: no setup, just the mic ==========================
        browser, page = await new_page(p, 420, 860)  # phone-sized
        await page.goto(f"{args.api}/")
        assert page.url.endswith("/voice/"), page.url
        await page.wait_for_selector("#mic:not([disabled])", timeout=20_000)
        await page.screenshot(path=str(shots / "user-1-start.png"))
        t0 = time.perf_counter()
        await page.click("#mic")
        await page.wait_for_selector(".msg.bot:not(.wait)", timeout=240_000)
        await page.wait_for_selector("#mic.speaking", timeout=120_000)
        user = {
            "title": await page.text_content("#title"),
            "question_heard": await page.text_content(".msg.me"),
            "answer": (await page.text_content(".msg.bot")).strip(),
            "status_while_speaking": await page.text_content("#status"),
            "developer_details_visible": await page.locator(".cite, .sources, .timing, #api-key").count(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        await page.screenshot(path=str(shots / "user-2-answer.png"))
        await page.click("#mic")  # interrupt the spoken answer
        await page.wait_for_function(
            "!document.querySelector('#mic').classList.contains('speaking')", timeout=10_000
        )
        user["interrupted_status"] = await page.text_content("#status")
        await page.fill("#type-input", args.typed)
        await page.press("#type-input", "Enter")
        await page.wait_for_function(
            "document.querySelectorAll('.msg.bot:not(.wait)').length >= 2", timeout=240_000
        )
        user["typed_answer"] = (await page.locator(".msg.bot").nth(1).text_content()).strip()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(shots / "user-3-conversation.png"))
        await browser.close()
        report["user_page"] = user

        # =========================== developer console ==========================================
        browser, page = await new_page(p, 1100, 900)
        await page.goto(f"{args.api}/voice/console.html")
        await page.fill("#api-key", api_key)
        await page.wait_for_function(
            "document.querySelector('#kb').options.length > 0 && !document.querySelector('#kb').disabled"
        )
        await page.select_option("#kb", label=args.kb)
        await page.fill("#vocab", args.vocab)
        await page.click("#connect-btn")
        await page.wait_for_selector("#talk:not([hidden])", timeout=20_000)
        t0 = time.perf_counter()
        await page.click("#mic")
        await page.wait_for_selector(".turn .a:not(.pending)", timeout=240_000)
        await page.wait_for_selector(".turn .replay:not([hidden])", timeout=120_000)
        dev = {
            "question_heard": await page.text_content(".turn .q-text"),
            "answer": (await page.text_content(".turn .a-text")).strip(),
            "sources": await page.locator(".turn .sources details").count(),
            "timing": (await page.text_content(".turn .legend")).strip(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        await page.click(".turn .inspect")
        await page.wait_for_selector("#trace-body table tbody tr", timeout=20_000)
        dev["trace_rows"] = await page.locator("#trace-body tbody tr").count()
        await page.screenshot(path=str(shots / "console-trace.png"))
        await browser.close()
        report["developer_console"] = dev

    report["console_errors"] = errors
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"screenshots: {shots}")
    if errors:
        raise SystemExit("browser console reported errors")


if __name__ == "__main__":
    asyncio.run(main())

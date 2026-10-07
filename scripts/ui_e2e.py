"""Browser test of the Voice RAG Console (/voice/) - a real Edge/Chromium with a fake microphone.

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
    ap.add_argument("--tenant-slug", required=True)
    ap.add_argument("--kb", required=True)
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

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel=None if args.channel == "chromium" else args.channel,
            headless=not args.headed,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={wav}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        page = await browser.new_page(viewport={"width": 1100, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        await page.goto(f"{args.api}/")
        assert page.url.endswith("/voice/"), page.url
        await page.fill("#api-key", api_key)
        await page.wait_for_function(
            "document.querySelector('#kb').options.length > 0 && !document.querySelector('#kb').disabled"
        )
        await page.select_option("#kb", label=args.kb)
        await page.fill("#vocab", args.vocab)
        await page.screenshot(path=str(shots / "1-setup.png"))
        await page.click("#connect-btn")
        await page.wait_for_selector("#talk:not([hidden])", timeout=20_000)
        pill = await page.text_content("#conn-pill")
        t0 = time.perf_counter()
        await page.click("#mic")  # start listening; the fake mic speaks after its lead-in

        # ---- spoken turn ------------------------------------------------------------------
        await page.wait_for_selector(".turn .a:not(.pending)", timeout=240_000)
        await page.wait_for_selector(".turn .replay:not([hidden])", timeout=120_000)
        spoken = {
            "connected": pill,
            "question_heard": await page.text_content(".turn .q-text"),
            "question_meta": await page.text_content(".turn .q-meta"),
            "answer": (await page.text_content(".turn .a-text")).strip(),
            "sources": await page.locator(".turn .sources details").count(),
            "citation_chips": await page.locator(".turn .cite").count(),
            "timing": (await page.text_content(".turn .legend")).strip(),
            "steps": await page.eval_on_selector_all(
                "#steps li", "els => els.map(e => e.className + ':' + e.innerText.replace(/\\n/g,' '))"
            ),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        await page.screenshot(path=str(shots / "2-spoken-answer.png"), full_page=True)

        # ---- trace inspector --------------------------------------------------------------
        await page.click(".turn .inspect")
        await page.wait_for_selector("#trace-body table tbody tr", timeout=20_000)
        trace = {
            "rows": await page.locator("#trace-body tbody tr").count(),
            "picked_rows": await page.locator("#trace-body tbody tr.picked").count(),
        }
        await page.screenshot(path=str(shots / "3-trace.png"))
        await page.click("#trace-close")

        # ---- citation chip opens its source -----------------------------------------------
        if spoken["citation_chips"]:
            await page.locator(".turn .cite").first.click()
            spoken["chip_opens_source"] = await page.locator(".turn .sources details[open]").count() > 0

        # ---- typed follow-up --------------------------------------------------------------
        await page.fill("#ask-input", args.typed)
        await page.click("#ask-form button")
        await page.wait_for_function(
            "document.querySelectorAll('.turn .a:not(.pending)').length >= 2", timeout=240_000
        )
        typed = {
            "question": await page.locator(".turn .q-text").nth(1).text_content(),
            "answer": (await page.locator(".turn .a-text").nth(1).text_content()).strip(),
            "sources": await page.locator(".turn").nth(1).locator(".sources details").count(),
        }
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(shots / "4-conversation.png"), full_page=True)
        await browser.close()

    print(
        json.dumps(
            {"spoken_turn": spoken, "trace": trace, "typed_turn": typed, "console_errors": errors}, indent=2
        )
    )
    print(f"screenshots: {shots}")
    if errors:
        raise SystemExit("browser console reported errors")


if __name__ == "__main__":
    asyncio.run(main())

"""One voice conversation: streamed PCM -> Silero VAD -> STT -> hybrid RAG -> TTS -> client.

Mirrors the Leap voice agent's turn-taking (server-side VAD, echo hold, barge-in, reliability
gate) with the hybrid RAG orchestrator as the "brain". Each turn runs as its own task so a
barge-in or reset can cancel it while retrieval/generation is still in flight.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.config import Settings
from app.providers.stt.base import SpeechToText, Transcript
from app.providers.tts.base import TextToSpeech
from app.rag.orchestrator import QueryOptions, RAGOrchestrator, RAGResult
from app.voice import messages as M
from app.voice.speech import for_speech, split_for_tts, to_iso_language
from app.voice.vad import VadStream

logger = logging.getLogger(__name__)

# Parallel TTS requests per answer (order of playback is preserved regardless).
TTS_CONCURRENCY = 4

Send = Callable[[list[dict[str, Any]]], Awaitable[None]]


class VoicePipeline:
    """Stateless STT -> RAG -> TTS steps, shared by the WebSocket session and REST endpoints."""

    def __init__(
        self, settings: Settings, orchestrator: RAGOrchestrator, stt: SpeechToText, tts: TextToSpeech
    ):
        self.settings = settings
        self.voice = settings.voice
        self.orchestrator = orchestrator
        self.stt = stt
        self.tts = tts

    async def transcribe(
        self, audio: bytes, filename: str = "utterance.wav", prompt: str | None = None
    ) -> tuple[Transcript, bool]:
        t = await self.stt.transcribe(
            audio, filename=filename, language=self.voice.stt_language, prompt=prompt or self.voice.stt_prompt
        )
        return t, t.is_reliable(self.voice.stt_min_avg_logprob, self.voice.stt_max_no_speech_prob)

    async def ask(
        self,
        *,
        tenant_id: uuid.UUID,
        kb_id: uuid.UUID,
        question: str,
        filters: dict | None,
        top_k: int | None,
    ) -> RAGResult:
        options = QueryOptions.from_settings(self.settings.retrieval, top_k=top_k)
        return await self.orchestrator.query(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            query=question,
            filters=filters or {},
            options=options,
        )

    def speech_language(self, transcript: Transcript | None) -> str:
        return to_iso_language(transcript.language if transcript else None) or self.voice.tts_language

    async def speak(self, text: str, language: str) -> bytes:
        if not self.tts.enabled or not text:
            return b""
        return await self.tts.synthesize(text, language=language)


class VoiceSession:
    def __init__(
        self,
        pipeline: VoicePipeline,
        send: Send,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        filters: dict | None = None,
        top_k: int | None = None,
        stt_prompt: str | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.stt_prompt = stt_prompt
        self.pipeline = pipeline
        self.send = send
        self.tenant_id = tenant_id
        self.kb_id = knowledge_base_id
        self.filters = filters or {}
        self.top_k = top_k
        # One VAD per connection: it carries LSTM state for this speaker's stream.
        self.vad = VadStream(pipeline.voice.vad)
        self._turn: asyncio.Task | None = None

    def client_config(self) -> dict[str, Any]:
        v = self.pipeline.voice
        return {
            "sample_rate": v.vad.sample_rate,
            "frame_samples": v.vad.frame_samples,
            "echo_hold_ms": v.echo_hold_ms,
            "stt": {"provider": self.pipeline.stt.provider, "model": self.pipeline.stt.model},
            "tts": {
                "enabled": self.pipeline.tts.enabled,
                "provider": self.pipeline.tts.provider,
                "voice": self.pipeline.tts.voice,
                "mime": self.pipeline.tts.mime_type,
            },
        }

    # --- inputs -----------------------------------------------------------------------------
    async def on_audio(self, pcm: bytes) -> None:
        for event in self.vad.feed(pcm):
            if event.is_start:
                await self.send([M.speech_started()])
            else:
                await self.send([M.speech_ended(event.duration_ms)])
                assert event.wav is not None
                self._start_turn(functools.partial(self._audio_turn, event.wav, "utterance.wav"))

    async def on_utterance(self, audio: bytes, filename: str = "utterance.wav") -> None:
        self.vad.reset()
        self._start_turn(lambda: self._audio_turn(audio, filename))

    async def on_text(self, text: str) -> None:
        self.vad.reset()  # anything captured so far belongs to a turn that is gone
        self._start_turn(lambda: self._question_turn(text, None))

    def hold(self, ms: int) -> None:
        self.vad.hold(ms)

    def drop_audio(self) -> None:
        self.vad.reset()

    async def barge_in(self) -> None:
        await self._cancel_turn()
        self.vad.reset()
        await self.send([M.stop_audio(), M.listen("barge_in")])

    async def reset(self) -> None:
        await self._cancel_turn()
        self.vad.reset()
        await self.send([M.stop_audio(), M.listen("reset")])

    async def close(self) -> None:
        await self._cancel_turn()

    # --- turns ------------------------------------------------------------------------------
    def _start_turn(self, make_turn: Callable[[], Awaitable[None]]) -> None:
        # Newest question wins: a fresh utterance supersedes one still being answered.
        # The coroutine is created inside the task, so a turn cancelled before it starts
        # leaves no un-awaited coroutine behind.
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
        self._turn = asyncio.create_task(self._guarded(make_turn), name=f"voice-turn-{self.id}")

    async def _cancel_turn(self) -> None:
        task, self._turn = self._turn, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _guarded(self, make_turn: Callable[[], Awaitable[None]]) -> None:
        try:
            await make_turn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("voice turn failed", extra={"session": self.id})
            code = getattr(exc, "code", "voice_turn_failed")
            await self.send([M.error(code, getattr(exc, "message", str(exc))[:300]), M.listen("error")])

    async def _audio_turn(self, audio: bytes, filename: str) -> None:
        await self.send([M.thinking()])
        t0 = time.perf_counter()
        transcript, reliable = await self.pipeline.transcribe(audio, filename, self.stt_prompt)
        stt_ms = round((time.perf_counter() - t0) * 1000, 1)
        await self.send(
            [
                M.transcript(
                    transcript.text, language=transcript.language, reliable=reliable, latency_ms=stt_ms
                )
            ]
        )
        if not reliable:
            # Unintelligible audio (noise, echo, a cough): reopen the mic without speaking, so
            # an apology can't echo back into the next false turn (Leap's silent reprompt).
            await self.send([M.listen("unintelligible")])
            return
        await self._question_turn(transcript.text, transcript, announce=False)

    async def _question_turn(
        self, question: str, transcript: Transcript | None, announce: bool = True
    ) -> None:
        if announce:
            await self.send([M.transcript(question, language=None, reliable=True), M.thinking()])
        t0 = time.perf_counter()
        result = await self.pipeline.ask(
            tenant_id=self.tenant_id,
            kb_id=self.kb_id,
            question=question,
            filters=self.filters,
            top_k=self.top_k,
        )
        rag_ms = round((time.perf_counter() - t0) * 1000, 1)
        answer_text = result.answer or ""
        await self.send(
            [
                M.answer(
                    answer_text,
                    [c.to_dict() for c in result.citations],
                    str(result.trace_id),
                    result.counts,
                    rag_ms,
                )
            ]
        )
        language = self.pipeline.speech_language(transcript)
        spoken = for_speech(answer_text, self.pipeline.voice.max_spoken_chars)
        await self._speak_streamed(spoken, language)
        await self.send([M.listen()])

    async def _speak_streamed(self, spoken: str, language: str) -> None:
        """Synthesize sentence-sized pieces concurrently and send each, in order, as soon as it
        is ready - the first sentence plays while the rest of the answer is still being voiced."""
        tts = self.pipeline.tts
        v = self.pipeline.voice
        pieces = (
            split_for_tts(spoken, v.tts_chunk_chars, v.tts_first_chunk_chars)
            if tts.enabled and spoken
            else []
        )
        if not pieces:  # text-only mode
            await self.send([M.say(spoken, "", tts.mime_type, language=language)])
            return
        sem = asyncio.Semaphore(TTS_CONCURRENCY)

        async def synth(piece: str) -> bytes:
            async with sem:
                return await self.pipeline.speak(piece, language)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(synth(p)) for p in pieces]
        reported = False
        try:
            for piece, task in zip(pieces, tasks, strict=True):
                try:
                    audio = await task
                except Exception as exc:  # keep going: the text answer is already on screen
                    audio = b""
                    logger.warning("text-to-speech failed", extra={"session": self.id, "error": repr(exc)})
                    if not reported:
                        reported = True
                        await self.send(
                            [M.error("tts_failed", "Speech synthesis failed for part of the answer")]
                        )
                await self.send(
                    [
                        M.say(
                            piece,
                            base64.b64encode(audio).decode("ascii") if audio else "",
                            tts.mime_type,
                            language=language,
                            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                        )
                    ]
                )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

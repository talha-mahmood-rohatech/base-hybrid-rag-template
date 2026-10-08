"""Silero voice-activity detection (self-contained; imports nothing else from the app).

from app.voice.vad import VadStream
stream = VadStream()                    # one per connection
for event in stream.feed(pcm_bytes):    # raw 16 kHz mono PCM16LE
    if event.is_end:
        transcribe(event.wav)
"""

from app.voice.vad.config import VadConfig
from app.voice.vad.stream import SpeechEvent, SpeechEventKind, VadStream

__all__ = ["SpeechEvent", "SpeechEventKind", "VadConfig", "VadStream"]

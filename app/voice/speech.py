"""Turn a RAG answer (markdown + citation markers) into text a TTS engine should read aloud."""

from __future__ import annotations

import re
import unicodedata

_CITATION_RE = re.compile(r"\s*(?:\[\d{1,3}(?:\s*,\s*\d{1,3})*\]|【[^】]*】)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_CODE_RE = re.compile(r"`{1,3}([^`]*)`{1,3}")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+", re.M)
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}.*$", re.M)
GROWTH = 1.6
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Whisper reports the language as a full English name; TTS engines want ISO codes.
_LANGUAGE_NAMES = {
    "english": "en",
    "urdu": "ur",
    "arabic": "ar",
    "french": "fr",
    "swahili": "sw",
    "hindi": "hi",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "turkish": "tr",
    "persian": "fa",
    "punjabi": "pa",
    "bengali": "bn",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "indonesian": "id",
    "malay": "ms",
    "dutch": "nl",
}


def to_iso_language(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip().lower()
    if len(raw) == 2:
        return raw
    return _LANGUAGE_NAMES.get(raw)


def for_speech(text: str, max_chars: int = 1200) -> str:
    """Strip citations/markdown, give list items a spoken pause, cut at a sentence boundary."""
    if not text:
        return ""
    # NFKC folds the narrow/no-break spaces some LLMs put inside numbers (e.g. U+202F).
    s = _CITATION_RE.sub("", unicodedata.normalize("NFKC", text))
    s = _IMAGE_RE.sub("", s)
    s = _LINK_RE.sub(r"\1", s)
    s = _CODE_RE.sub(r"\1", s)
    s = _EMPHASIS_RE.sub(lambda m: m.group(2) or m.group(3) or "", s)
    s = _TABLE_RULE_RE.sub("", s)
    s = _HEADING_RE.sub("", s)
    s = _BULLET_RE.sub("", s)

    lines = []
    for raw in s.splitlines():
        line = raw.replace("|", ", ").strip(" ,")
        if not line:
            continue
        # A list item or heading without punctuation would run straight into the next line.
        if line[-1] not in ".!?:;,":
            line += "."
        lines.append(line)
    s = " ".join(lines)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    if len(s) <= max_chars:
        return s
    out = ""
    for sentence in _SENTENCE_END_RE.split(s):
        if len(out) + len(sentence) + 1 > max_chars:
            break
        out = f"{out} {sentence}".strip()
    return out or s[:max_chars].rsplit(" ", 1)[0] + "."


def split_for_tts(text: str, max_chars: int = 200, first_max_chars: int | None = None) -> list[str]:
    """Group sentences into speakable pieces.

    The first piece is kept short (``first_max_chars``) because synthesis runs at roughly real
    time: the listener hears audio once that piece is voiced. Later pieces grow by ``GROWTH``
    up to ``max_chars`` and are synthesized in parallel while earlier ones play. Over-long sentences are split at
    commas, then at spaces.
    """
    first_max = min(first_max_chars or max_chars, max_chars)
    atoms: list[str] = []
    for sentence in _SENTENCE_END_RE.split(text.strip()):
        if len(sentence) <= first_max:
            atoms.append(sentence)
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sentence):
            while len(clause) > first_max:
                cut = clause.rfind(" ", 0, first_max)
                cut = cut if cut > 0 else first_max
                atoms.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            atoms.append(clause)

    pieces: list[str] = []
    current = ""
    for atom in (a for a in atoms if a):
        # Pieces grow geometrically: with all pieces synthesized in parallel at ~real time, each
        # one is ready before playback of everything ahead of it finishes, so there are no gaps.
        limit = min(max_chars, int(first_max * GROWTH ** len(pieces)))
        if current and len(current) + 1 + len(atom) > limit:
            pieces.append(current)
            current = atom
        else:
            current = f"{current} {atom}".strip()
    if current:
        pieces.append(current)
    return pieces

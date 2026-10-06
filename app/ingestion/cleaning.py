"""Text normalization applied to every segment before chunking."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_SPACES_RE = re.compile(r"[ \t\u00a0]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")


def clean_text(text: str, *, fix_hyphenation: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    if fix_hyphenation:  # PDF line-wrap hyphenation: "exam-\nple" -> "example"
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _SPACES_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()

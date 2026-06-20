# -*- coding: utf-8 -*-
"""Dependency-free Bulgarian text normalization shared by MFA and inference.

This module intentionally lives outside the :mod:`text` package.  Alignment is
performed before the FastSpeech2 phone inventory is loaded, so importing the
normalizer must not import ``text.symbols`` as a side effect.
"""

import re
from typing import List, Tuple

from num2wordBg import text_numbers_to_words


BG_ALPHABET = tuple("абвгдежзийклмнопрстуфхцчшщъьюя")

_NORMALIZE = {
    "ѝ": "и",
    "ё": "е",
    "ы": "и",
}
_VALID = set(BG_ALPHABET)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_PAUSE_RE = re.compile(r"[,;:.!?]+")


def normalize_words(text: str, expand_numbers: bool = True) -> str:
    """Return lowercase Bulgarian words separated by single spaces.

    Punctuation and unsupported scripts become word boundaries.  Number
    expansion is enabled by default so the exact same function can be used for
    MFA ``.lab`` files and raw-text synthesis.
    """
    if expand_numbers:
        text = text_numbers_to_words(text)

    out = []
    for ch in text.lower():
        ch = _NORMALIZE.get(ch, ch)
        out.append(ch if ch in _VALID else " ")

    cleaned = _WHITESPACE_RE.sub(" ", "".join(out)).strip()
    return _REPEAT_RE.sub(r"\1\1", cleaned)


def normalize_for_mfa(text: str) -> str:
    """Canonical orthographic transcript written to MFA ``.lab`` files."""
    return normalize_words(text, expand_numbers=True)


def synthesis_segments(text: str) -> List[Tuple[str, bool]]:
    """Normalize raw synthesis text while retaining punctuation pauses.

    Returns ``[(normalized_words, pause_after), ...]``.  The caller converts a
    pause between two non-empty segments to the canonical ``sp`` phone.  Final
    punctuation does not create a trailing token because FastSpeech2 trims
    leading/trailing silence during training.
    """
    expanded = text_numbers_to_words(text)
    raw_parts = _PAUSE_RE.split(expanded)
    separators = _PAUSE_RE.findall(expanded)

    normalized = [normalize_words(part, expand_numbers=False) for part in raw_parts]
    result = []
    for index, words in enumerate(normalized):
        if not words:
            continue
        later_has_words = any(normalized[index + 1 :])
        pause_after = index < len(separators) and later_has_words
        result.append((words, pause_after))
    return result


def foreign_letters(text: str):
    """Return alphabetic characters that are not Bulgarian after folding."""
    bad = set()
    for ch in text.lower():
        ch = _NORMALIZE.get(ch, ch)
        if ch.isalpha() and ch not in _VALID:
            bad.add(ch)
    return bad


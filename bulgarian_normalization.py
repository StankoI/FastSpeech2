# -*- coding: utf-8 -*-
"""Shared Bulgarian normalization for alignment, training, and synthesis.

MFA must receive words only, while FastSpeech2 also needs the prosodic marks
that were present in the source transcript.  Both representations are derived
from :func:`prosody_words`; this prevents train/inference normalization drift.
"""

import re
from typing import List, Optional, Tuple

from num2wordBg import text_numbers_to_words


NORMALIZER_VERSION = 2
BG_ALPHABET = tuple("абвгдежзийклмнопрстуфхцчшщъьюя")

PUNCTUATION_TO_TOKEN = {
    ",": "p_comma",
    ";": "p_semicolon",
    ":": "p_colon",
    ".": "p_period",
    "?": "p_question",
    "!": "p_exclamation",
    "—": "p_dash",
}
TOKEN_TO_PUNCTUATION = {token: mark for mark, token in PUNCTUATION_TO_TOKEN.items()}
PUNCTUATION_TOKENS = tuple(PUNCTUATION_TO_TOKEN.values())

_NORMALIZE = {"ѝ": "и", "ё": "е", "ы": "и"}
_VALID = set(BG_ALPHABET)
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_DECIMAL_SEPARATOR_RE = re.compile(r"(?<=\d)[,.](?=\d)")
_SPACED_HYPHEN_RE = re.compile(r"(?<=\s)-(?=\s)")
_DASHES = {"–", "—", "―"}
_PLACEHOLDER_PREFIX = "__bgpunct_"


def _expand_preserving_punctuation(text: str) -> str:
    """Expand numbers without turning decimal separators into pause marks."""
    text = _DECIMAL_SEPARATOR_RE.sub("§", text)
    text = _SPACED_HYPHEN_RE.sub("—", text)
    protected = []
    for ch in text:
        ch = "—" if ch in _DASHES else ch
        token = PUNCTUATION_TO_TOKEN.get(ch)
        if token:
            protected.append(" {}{}__ ".format(_PLACEHOLDER_PREFIX, token[2:]))
        else:
            protected.append(ch)
    return text_numbers_to_words("".join(protected))


def _choose_punctuation(tokens: List[str]) -> Optional[str]:
    """Collapse a punctuation run such as ``?!`` to one stable control token."""
    if not tokens:
        return None
    for preferred in ("p_question", "p_exclamation", "p_period"):
        if preferred in tokens:
            return preferred
    return tokens[-1]


def prosody_words(text: str, expand_numbers: bool = True) -> List[Tuple[str, Optional[str]]]:
    """Return ordered ``(word, punctuation_after_word)`` pairs.

    Unsupported characters separate words. Quotes and brackets are discarded;
    the supported punctuation marks become explicit FastSpeech2 control tokens.
    """
    expanded = _expand_preserving_punctuation(text) if expand_numbers else text
    pieces = re.split(
        r"(__bgpunct_(?:comma|semicolon|colon|period|question|exclamation|dash)__)",
        expanded.lower(),
    )
    result: List[List[Optional[str]]] = []
    pending: List[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.startswith(_PLACEHOLDER_PREFIX) and piece.endswith("__"):
            token = piece[len(_PLACEHOLDER_PREFIX) : -2]
            token = "p_" + token
            if result:
                pending.append(token)
            continue

        # ``expand_numbers=False`` is also a public path, so recognize literal
        # punctuation here rather than only protected placeholders.
        chunk = []
        for ch in piece:
            ch = _NORMALIZE.get(ch, ch)
            if ch in _DASHES:
                ch = "—"
            punct = PUNCTUATION_TO_TOKEN.get(ch)
            if punct:
                if chunk:
                    words = _words_from_chars(chunk)
                    for word in words:
                        if pending and result:
                            result[-1][1] = _choose_punctuation(pending)
                            pending = []
                        result.append([word, None])
                    chunk = []
                if result:
                    pending.append(punct)
            else:
                chunk.append(ch)
        for word in _words_from_chars(chunk):
            if pending and result:
                result[-1][1] = _choose_punctuation(pending)
                pending = []
            result.append([word, None])

    if pending and result:
        result[-1][1] = _choose_punctuation(pending)
    return [(str(word), token) for word, token in result]


def _words_from_chars(chars) -> List[str]:
    folded = []
    for ch in chars:
        ch = _NORMALIZE.get(ch, ch)
        folded.append(ch if ch in _VALID else " ")
    return [
        _REPEAT_RE.sub(r"\1\1", word)
        for word in "".join(folded).split()
        if word
    ]


def normalize_words(text: str, expand_numbers: bool = True) -> str:
    """Return the canonical word-only representation used by MFA."""
    return " ".join(word for word, _ in prosody_words(text, expand_numbers))


def normalize_for_mfa(text: str) -> str:
    return normalize_words(text, expand_numbers=True)


def normalize_with_punctuation(text: str) -> str:
    """Return normalized words with supported source punctuation preserved."""
    return render_prosody_words(prosody_words(text))


def render_prosody_words(pairs) -> str:
    """Render already-normalized ``(word, token)`` pairs without reprocessing."""
    chunks = []
    for word, token in pairs:
        chunks.append(word)
        if token:
            mark = TOKEN_TO_PUNCTUATION[token]
            if token == "p_dash":
                chunks.append(mark)
            else:
                chunks[-1] += mark
    return " ".join(chunks)


def synthesis_segments(text: str):
    """Backward-compatible view; new code should use :func:`prosody_words`."""
    pairs = prosody_words(text)
    return [(word, token is not None) for word, token in pairs]


def foreign_letters(text: str):
    bad = set()
    for ch in text.lower():
        ch = _NORMALIZE.get(ch, ch)
        if ch.isalpha() and ch not in _VALID:
            bad.add(ch)
    return bad

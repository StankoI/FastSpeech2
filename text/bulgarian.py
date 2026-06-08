# -*- coding: utf-8 -*-
"""Grapheme-based G2P for Bulgarian.

Bulgarian orthography is highly phonemic, so instead of a phoneme dictionary we
treat each Cyrillic letter as a token ("phone"). This module is the single
source of truth for that mapping and is used in three places:

  * building the MFA pronunciation dictionary (word -> space-separated letters),
  * defining the symbol set in ``text/symbols.py``,
  * converting input text to tokens at synthesis time.

Keep this list and ``word_to_phonemes`` in sync: every token the function can
emit must be registered in ``symbols.py``.
"""

import re

# The 30 letters of the modern Bulgarian alphabet (lowercase).
BG_ALPHABET = list("абвгдежзийклмнопрстуфхцчшщъьюя")

# Characters that occur in the corpus but should be folded onto a canonical
# Bulgarian letter before tokenisation (accented / foreign look-alikes that are
# acoustically equivalent). Anything not here and not in BG_ALPHABET is dropped.
_NORMALIZE = {
    "ѝ": "и",  # и with grave accent (U+045D), the word "ѝ" -> и
    "ё": "е",  # Russian "yo" -> е
    "ы": "и",  # Russian "yery" -> closest Bulgarian sound и
}

_VALID = set(BG_ALPHABET)

_WHITESPACE_RE = re.compile(r"\s+")
# Bulgarian never has 3+ identical letters in a row; runs that long are
# transcription artifacts (drawn-out vowels/screams). Collapse them to 2.
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def normalize_text(text):
    """Lowercase, fold look-alikes, and keep only Bulgarian letters and single
    spaces. Punctuation, digits and any other characters become spaces; runs of
    whitespace collapse to one, and runs of 3+ identical letters collapse to 2.
    Returns the cleaned word string (may be empty).

    Numbers should already have been expanded to words upstream (see
    ``num2wordBg.text_numbers_to_words``).

    >>> normalize_text("Здравей, свят!")
    'здравей свят'
    >>> normalize_text("«Шибил».")
    'шибил'
    >>> normalize_text("еххххх")
    'ехх'
    """
    out = []
    for ch in text.lower():
        ch = _NORMALIZE.get(ch, ch)
        out.append(ch if ch in _VALID else " ")
    cleaned = _WHITESPACE_RE.sub(" ", "".join(out)).strip()
    return _REPEAT_RE.sub(r"\1\1", cleaned)


def foreign_letters(text):
    """Return the set of alphabetic characters in *text* that are not Bulgarian
    (after case-folding and normalisation). An empty set means the text is pure
    Bulgarian; a non-empty set flags foreign/garbled lines worth dropping.

    >>> sorted(foreign_letters("Записано за LibriVox"))
    ['b', 'i', 'l', 'o', 'r', 'v', 'x']
    >>> foreign_letters("чисто български")
    set()
    """
    bad = set()
    for ch in text.lower():
        ch = _NORMALIZE.get(ch, ch)
        if ch.isalpha() and ch not in _VALID:
            bad.add(ch)
    return bad


def word_to_phonemes(word):
    """Split a single word into a list of Bulgarian letter-tokens.

    Lowercases the word, folds accented/foreign look-alikes onto canonical
    letters, and drops anything that is not a Bulgarian letter (punctuation,
    digits, stray Latin/other-script characters). Numbers should already have
    been expanded to words upstream (see ``num2wordBg.text_numbers_to_words``).

    Returns an empty list if nothing usable remains.

    >>> word_to_phonemes("Здравей")
    ['з', 'д', 'р', 'а', 'в', 'е', 'й']
    >>> word_to_phonemes("LibriVox")
    []
    >>> word_to_phonemes("ѝ")
    ['и']
    """
    tokens = []
    for ch in word.lower():
        ch = _NORMALIZE.get(ch, ch)
        if ch in _VALID:
            tokens.append(ch)
    return tokens

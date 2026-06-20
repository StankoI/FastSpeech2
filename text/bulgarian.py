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

# The grapheme helpers below are retained for legacy scripts only.  The phoneme
# pipeline uses the dependency-free canonical normalizer in
# ``bulgarian_normalization.py``.
from bulgarian_normalization import (
    BG_ALPHABET,
    foreign_letters,
    normalize_words as normalize_text,
)

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

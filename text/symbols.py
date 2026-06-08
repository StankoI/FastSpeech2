""" adapted from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

For Bulgarian we use grapheme tokens: one token per Cyrillic letter, prefixed
with "@" to match the curly-brace path in text/__init__.py. See text/bulgarian.py.
"""

from text import bulgarian

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Bulgarian graphemes (each Cyrillic letter is a token); see text/bulgarian.py.
_bulgarian = ["@" + s for s in bulgarian.BG_ALPHABET]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _bulgarian
    + _silences
)

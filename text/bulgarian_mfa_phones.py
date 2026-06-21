"""Frozen Bulgarian MFA phone inventory used by every phoneme checkpoint.

The 46 lexical phones are the canonical Bulgarian MFA/G2P inventory.  ``sp``
is the internal-pause token, ``sil`` is retained for compatibility with older
TextGrids, and ``spn`` is reserved for genuine unknown/noise spans.  Never
reorder this list for an existing checkpoint.
"""

INVENTORY_VERSION = 2

PHONES = [
    "a",
    "b",
    "bʲ",
    "c",
    "dʒ",
    "dʲ",
    "d̪",
    "f",
    "fʲ",
    "i",
    "j",
    "k",
    "m",
    "mʲ",
    "n̪",
    "o",
    "p",
    "pʲ",
    "r",
    "rʲ",
    "sʲ",
    "s̪",
    "tsʲ",
    "tʃ",
    "tʲ",
    "t̪",
    "t̪s̪",
    "u",
    "v",
    "vʲ",
    "x",
    "zʲ",
    "z̪",
    "ç",
    "ŋ",
    "ɔ",
    "ɛ",
    "ɟ",
    "ɡ",
    "ɤ",
    "ɫ",
    "ɱ",
    "ɲ",
    "ʃ",
    "ʎ",
    "ʒ",
    "sp",
    "sil",
    "spn",
    # Non-acoustic linguistic controls. ``wb`` always has zero target duration;
    # punctuation may carry an aligned pause duration or zero frames.
    "wb",
    "p_comma",
    "p_semicolon",
    "p_colon",
    "p_period",
    "p_question",
    "p_exclamation",
    "p_dash",
]

WORD_BOUNDARY_TOKEN = "wb"
PUNCTUATION_TOKENS = (
    "p_comma",
    "p_semicolon",
    "p_colon",
    "p_period",
    "p_question",
    "p_exclamation",
    "p_dash",
)
CONTROL_TOKENS = (WORD_BOUNDARY_TOKEN,) + PUNCTUATION_TOKENS

"""Frozen Bulgarian MFA phone inventory used by every phoneme checkpoint.

The 46 lexical phones are the canonical Bulgarian MFA/G2P inventory.  ``sp``
is the internal-pause token, ``sil`` is retained for compatibility with older
TextGrids, and ``spn`` is reserved for genuine unknown/noise spans.  Never
reorder this list for an existing checkpoint.
"""

INVENTORY_VERSION = 1

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
]


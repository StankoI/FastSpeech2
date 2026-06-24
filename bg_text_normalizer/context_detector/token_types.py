# -*- coding: utf-8 -*-
"""Token type definitions for Bulgarian text."""

from enum import Enum, auto


class TokenType(Enum):
    """Types of tokens that can be detected in Bulgarian text."""

    # Numbers
    CARDINAL = auto()       # Plain number: "15" → "петнадесет"
    ORDINAL = auto()        # Ordinal: "№15", "1-ви" → "първи"

    # Currency
    CURRENCY = auto()       # Amount + currency: "15 лв." → "петнадесет лева"

    # Date/Time
    DATE = auto()           # Date: "21.04.2026"
    TIME = auto()           # Time: "14:30" (future)

    # Contact
    PHONE = auto()          # Phone: "0888123456"

    # Measurements
    MEASUREMENT = auto()    # Number + unit: "3 кг" → "три килограма"

    # Text
    WORD = auto()           # Regular word
    ABBREVIATION = auto()   # Abbreviation: "г.", "ул."

    # Punctuation
    PUNCTUATION = auto()    # . , ! ? ; : - etc.

    # Other
    SYMBOL = auto()         # Other symbols
    WHITESPACE = auto()     # Spaces, tabs
    UNKNOWN = auto()        # Cannot determine

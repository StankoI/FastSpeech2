# -*- coding: utf-8 -*-
"""Regex patterns for Bulgarian text tokenization."""

import re

# Date patterns: DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY
DATE_PATTERN = re.compile(
    r'\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b'
)

# Phone patterns: Bulgarian mobile (08XXXXXXXX) and international (+359XXXXXXXXX)
PHONE_PATTERN = re.compile(
    r'\b(?:\+359|0)[\s\-]?\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3,4}\b'
)

# Currency patterns: number followed by currency symbol
CURRENCY_MARKERS = re.compile(
    r'\b(лв\.?|лева|евро|EUR|USD|\$|€)\b',
    re.IGNORECASE
)

# Measurement units
UNIT_MARKERS = re.compile(
    r'\b(кг|г|мг|т|м|см|мм|км|л|мл|кв\.?\s*м|куб\.?\s*м)\b',
    re.IGNORECASE
)

# Ordinal markers: №, #, or suffix like -ви, -ва, -во
ORDINAL_PATTERN = re.compile(
    r'(?:№|#)\s*\d+|\d+[\-]?(?:ви|ва|во|ри|ра|ро|ти|та|то|ми|ма|мо)\b',
    re.IGNORECASE
)

# Number pattern: integers and decimals
NUMBER_PATTERN = re.compile(
    r'\b\d+(?:[,\.]\d+)?\b'
)

# Word pattern: Cyrillic and Latin letters
WORD_PATTERN = re.compile(
    r'[а-яА-ЯёЁa-zA-Z]+(?:[\-][а-яА-ЯёЁa-zA-Z]+)*'
)

# Punctuation
PUNCTUATION_PATTERN = re.compile(
    r'[.,!?;:\-\—\–\(\)\[\]\{\}\"\'«»„"…]'
)

# Combined tokenizer pattern - order matters!
TOKEN_PATTERN = re.compile(
    r'(?P<date>\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})'  # Dates first (most specific)
    r'|(?P<phone>(?:\+359|0)[\s\-]?\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3,4})'  # Phones
    r'|(?P<ordinal>(?:№|#)\s*\d+|\d+[\-]?(?:ви|ва|во|ри|ра|ро|ти|та|то|ми|ма|мо))'  # Ordinals
    r'|(?P<number>\d+(?:[,\.]\d+)?)'  # Numbers
    r'|(?P<word>[а-яА-ЯёЁa-zA-Z]+(?:[\-][а-яА-ЯёЁa-zA-Z]+)*\.?)'  # Words (with optional trailing dot for abbrev)
    r'|(?P<punctuation>[.,!?;:\-\—\–\(\)\[\]\{\}\"\'«»„"…])'  # Punctuation
    r'|(?P<symbol>[^\s\w])'  # Other symbols
    r'|(?P<whitespace>\s+)',  # Whitespace
    re.IGNORECASE | re.UNICODE
)

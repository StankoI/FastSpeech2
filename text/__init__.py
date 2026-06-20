"""Text/phone conversion with strict validation for phoneme sequences."""

import re
import unicodedata

from text import cleaners
from text.symbols import symbols


_symbol_to_id = {symbol: index for index, symbol in enumerate(symbols)}
_id_to_symbol = {index: symbol for index, symbol in enumerate(symbols)}
_curly_re = re.compile(r"(.*?)\{(.*?)\}(.*)")


def text_to_sequence(text, cleaner_names):
    """Convert text to IDs; braced phone sequences are validated strictly."""
    sequence = []
    while text:
        match = _curly_re.match(text)
        if not match:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(match.group(1), cleaner_names))
        sequence += phones_to_sequence(match.group(2))
        text = match.group(3)
    return sequence


def phones_to_sequence(phone_text):
    """Convert whitespace-separated MFA phones to IDs or raise on any unknown."""
    phones = [unicodedata.normalize("NFC", p) for p in phone_text.split()]
    if not phones:
        raise ValueError("Empty phoneme sequence is not valid FastSpeech2 input")

    tagged = ["@" + phone for phone in phones]
    unknown = [phone for phone, symbol in zip(phones, tagged) if symbol not in _symbol_to_id]
    if unknown:
        raise ValueError("Unknown MFA phone(s): {}".format(", ".join(sorted(set(unknown)))))
    return [_symbol_to_id[symbol] for symbol in tagged]


def sequence_to_text(sequence):
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            symbol = _id_to_symbol[symbol_id]
            if len(symbol) > 1 and symbol[0] == "@":
                symbol = "{%s}" % symbol[1:]
            result += symbol
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if cleaner is None:
            raise ValueError("Unknown cleaner: {}".format(name))
        text = cleaner(text)
    return text


def _symbols_to_sequence(raw_symbols):
    # Non-phone text is retained only for compatibility with the original API.
    # Bulgarian train/inference data must use the braced strict path above.
    return [
        _symbol_to_id[symbol]
        for symbol in raw_symbols
        if symbol in _symbol_to_id and symbol not in {"_", "~"}
    ]

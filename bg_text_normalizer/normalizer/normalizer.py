# -*- coding: utf-8 -*-
"""Main normalizer orchestrator for Bulgarian text."""

from dataclasses import dataclass
from typing import List, Optional

from ..tokenizer import Token, Tokenizer, tokenize, reconstruct
from ..context_detector import TokenType, DetectedToken, ContextDetector, detect
from ..numbers_to_words import number_to_words

from .currency import normalize_currency
from .dates import normalize_date_from_dict
from .measurements import normalize_measurement_from_dict
from .phones import normalize_phone
from .ordinals import normalize_ordinal_from_dict


@dataclass
class NormalizedResult:
    """Result of normalizing text."""
    original_text: str
    normalized_text: str
    tokens: List[DetectedToken]


def normalize(text: str) -> str:
    """
    Конвертира български текст в напълно произносима форма.

    Комбинира токенизация, детекция на контекст и нормализация.

    Args:
        text: Входен текст с числа, дати, съкращения и др.

    Returns:
        Текст с всички елементи конвертирани в произносим български

    Examples:
        >>> normalize("Сумата е 15 лв. на 21.04.2026")
        'Сумата е петнадесет лева на двадесет и първи април две хиляди двадесет и шеста година'
        >>> normalize("Тегло: 3 кг")
        'Тегло: три килограма'
        >>> normalize("Тел: 0888123456")
        'Тел: нула осем осем осем едно две три четири пет шест'
    """
    # Step 1: Tokenize
    tokens = tokenize(text)

    # Step 2: Detect context
    detected = detect(tokens)

    # Step 3: Normalize
    return normalize_tokens(detected)


def normalize_tokens(detected_tokens: List[DetectedToken]) -> str:
    """
    Нормализира вече детектирани токени.

    Args:
        detected_tokens: Списък от DetectedToken

    Returns:
        Нормализиран текст
    """
    parts = []

    for dt in detected_tokens:
        normalized = normalize_token(dt)
        parts.append(normalized)

        # Add whitespace after token
        if dt.whitespace_after:
            parts.append(dt.whitespace_after)

    return ''.join(parts)


def _normalize_decimal_cardinal(value: str) -> str:
    """Pronounce a plain decimal number without dropping the fractional part."""
    normalized = value.replace(",", ".")
    whole_text, _, fractional_text = normalized.partition(".")
    fractional_text = fractional_text.rstrip("0") or "0"

    whole = int(whole_text) if whole_text else 0
    if whole == 1:
        whole_word = "едно"
    elif whole == 2:
        whole_word = "две"
    else:
        whole_word = number_to_words(whole)

    fractional_words = " ".join(number_to_words(int(digit)) for digit in fractional_text)
    return f"{whole_word} цяло и {fractional_words}"


def normalize_token(detected: DetectedToken) -> str:
    """
    Нормализира единичен детектиран токен.

    Args:
        detected: DetectedToken обект

    Returns:
        Нормализиран текст за токена
    """
    token_type = detected.token_type
    metadata = detected.metadata
    value = detected.value

    if token_type == TokenType.WORD:
        return value

    if token_type == TokenType.CARDINAL:
        num_value = metadata.get('value', 0)
        if isinstance(num_value, int):
            return number_to_words(num_value)
        if isinstance(num_value, float) and not num_value.is_integer():
            return _normalize_decimal_cardinal(value)
        return number_to_words(int(num_value))

    if token_type == TokenType.ORDINAL:
        return normalize_ordinal_from_dict(metadata)

    if token_type == TokenType.CURRENCY:
        amount = metadata.get('amount', 0)
        currency = metadata.get('currency', 'BGN')
        return normalize_currency(amount, currency)

    if token_type == TokenType.DATE:
        return normalize_date_from_dict(metadata)

    if token_type == TokenType.PHONE:
        digits = metadata.get('digits', value)
        return normalize_phone(digits)

    if token_type == TokenType.MEASUREMENT:
        return normalize_measurement_from_dict(metadata)

    if token_type == TokenType.PUNCTUATION:
        return value

    if token_type == TokenType.ABBREVIATION:
        # Разширението идва от детектора (rules.ABBREVIATIONS); None означава
        # двусмислено/непознато съкращение — оставяме го както е.
        expansion = metadata.get('expansion')
        return expansion if expansion is not None else value

    if token_type == TokenType.WHITESPACE:
        return value

    # SYMBOL, UNKNOWN, etc.
    return value


class Normalizer:
    """Configurable text normalizer."""

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        detector: Optional[ContextDetector] = None
    ):
        self.tokenizer = tokenizer or Tokenizer()
        self.detector = detector or ContextDetector()

    def normalize(self, text: str) -> NormalizedResult:
        """Full normalization pipeline with detailed result."""
        tokens = self.tokenizer.tokenize(text)
        detected = self.detector.detect(tokens)
        normalized_text = normalize_tokens(detected)

        return NormalizedResult(
            original_text=text,
            normalized_text=normalized_text,
            tokens=detected
        )

    def normalize_text(self, text: str) -> str:
        """Simple normalization returning only the text."""
        return normalize(text)

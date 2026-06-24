# -*- coding: utf-8 -*-
"""Context detector for Bulgarian text tokens."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..tokenizer import Token
from .token_types import TokenType
from . import rules


@dataclass
class DetectedToken:
    """Token with detected context information."""
    token: Token
    token_type: TokenType
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized_value: Optional[str] = None  # Set by normalizer
    consumed: bool = False  # True if merged into another token

    @property
    def value(self) -> str:
        return self.token.value

    @property
    def whitespace_after(self) -> str:
        return self.token.whitespace_after


def detect(tokens: List[Token]) -> List[DetectedToken]:
    """
    Определя контекста и типа на всеки токен.

    Анализира последователности от токени за определяне на:
    - Валута: число + маркер (15 лв.)
    - Дати: DD.MM.YYYY
    - Телефони: български формат
    - Мерни единици: число + единица (3 кг)
    - Редни числителни: №15, 1-ви

    Args:
        tokens: Списък от токени

    Returns:
        Списък от DetectedToken с тип и метаданни
    """
    detected: List[DetectedToken] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else None

        # Skip already processed whitespace tokens
        if token.raw_type == "whitespace":
            detected.append(DetectedToken(token, TokenType.WHITESPACE))
            i += 1
            continue

        # Check for date (already identified by tokenizer)
        if token.raw_type == "date" or rules.is_date(token.value):
            date_info = rules.parse_date(token.value)
            detected.append(DetectedToken(
                token, TokenType.DATE, metadata=date_info
            ))
            i += 1
            continue

        # Check for phone
        if token.raw_type == "phone" or rules.is_phone(token.value):
            detected.append(DetectedToken(
                token, TokenType.PHONE,
                metadata={'digits': ''.join(c for c in token.value if c.isdigit())}
            ))
            i += 1
            continue

        # Check for ordinal (№15, 1-ви)
        if token.raw_type == "ordinal" or rules.is_ordinal(token.value):
            ordinal_info = rules.parse_ordinal(token.value)
            detected.append(DetectedToken(
                token, TokenType.ORDINAL, metadata=ordinal_info
            ))
            i += 1
            continue

        # Check for number followed by currency/unit
        if token.raw_type == "number" and next_token:
            next_value = next_token.value.lower().rstrip('.')

            # Check for currency: "15 лв."
            if rules.is_currency_marker(next_value) or rules.is_currency_marker(next_token.value):
                currency_code = rules.get_currency_code(next_value) or rules.get_currency_code(next_token.value)
                # Merge tokens
                merged_token = Token(
                    value=f"{token.value} {next_token.value}",
                    start=token.start,
                    end=next_token.end,
                    whitespace_after=next_token.whitespace_after,
                    raw_type="currency"
                )
                detected.append(DetectedToken(
                    merged_token, TokenType.CURRENCY,
                    metadata={
                        'amount': _parse_number(token.value),
                        'currency': currency_code,
                        'original_marker': next_token.value
                    }
                ))
                i += 2
                continue

            # Check for measurement: "3 кг"
            if rules.is_measurement_unit(next_value):
                singular, plural, gender = rules.get_unit_info(next_value)
                merged_token = Token(
                    value=f"{token.value} {next_token.value}",
                    start=token.start,
                    end=next_token.end,
                    whitespace_after=next_token.whitespace_after,
                    raw_type="measurement"
                )
                detected.append(DetectedToken(
                    merged_token, TokenType.MEASUREMENT,
                    metadata={
                        'value': _parse_number(token.value),
                        'unit': next_value,
                        'singular': singular,
                        'plural': plural,
                        'gender': gender
                    }
                ))
                i += 2
                continue

        # Plain number
        if token.raw_type == "number":
            detected.append(DetectedToken(
                token, TokenType.CARDINAL,
                metadata={'value': _parse_number(token.value)}
            ))
            i += 1
            continue

        # Word or abbreviation
        if token.raw_type == "word":
            if rules.is_abbreviation(token.value):
                expansion = rules.get_abbreviation_expansion(token.value)
                detected.append(DetectedToken(
                    token, TokenType.ABBREVIATION,
                    metadata={'expansion': expansion}
                ))
            else:
                detected.append(DetectedToken(token, TokenType.WORD))
            i += 1
            continue

        # Punctuation
        if token.raw_type == "punctuation":
            detected.append(DetectedToken(token, TokenType.PUNCTUATION))
            i += 1
            continue

        # Symbol or unknown
        detected.append(DetectedToken(
            token,
            TokenType.SYMBOL if token.raw_type == "symbol" else TokenType.UNKNOWN
        ))
        i += 1

    return detected


def _parse_number(text: str) -> float:
    """Parse number string, handling both . and , as decimal separator."""
    # Bulgarian often uses comma as decimal separator
    normalized = text.replace(',', '.')
    try:
        if '.' in normalized:
            return float(normalized)
        return int(normalized)
    except ValueError:
        return 0


class ContextDetector:
    """Stateful context detector."""

    def __init__(self):
        pass

    def detect(self, tokens: List[Token]) -> List[DetectedToken]:
        """Detect types for all tokens."""
        return detect(tokens)

    def detect_single(
        self,
        token: Token,
        prev_tokens: Optional[List[Token]] = None,
        next_tokens: Optional[List[Token]] = None
    ) -> DetectedToken:
        """Detect type for a single token with context."""
        # Build context list
        context = []
        if prev_tokens:
            context.extend(prev_tokens)
        context.append(token)
        if next_tokens:
            context.extend(next_tokens)

        # Detect all and find our token
        detected = detect(context)
        for d in detected:
            if d.token.start == token.start and d.token.end == token.end:
                return d

        return DetectedToken(token, TokenType.UNKNOWN)

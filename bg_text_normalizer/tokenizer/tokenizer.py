# -*- coding: utf-8 -*-
"""Tokenizer for Bulgarian text."""

from dataclasses import dataclass
from typing import List, Iterator
from .patterns import TOKEN_PATTERN


@dataclass
class Token:
    """Represents a token extracted from text."""
    value: str
    start: int
    end: int
    whitespace_after: str = ""
    raw_type: str = "unknown"  # Type hint from tokenizer (date, number, word, etc.)

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"Token({self.value!r}, {self.start}, {self.end}, ws={self.whitespace_after!r})"


def tokenize(text: str, preserve_whitespace: bool = False) -> List[Token]:
    """
    Разбива българския текст на токени.

    Args:
        text: Входен текст
        preserve_whitespace: Ако е True, включва whitespace като отделни токени

    Returns:
        Списък от Token обекти със запазена позиция

    Example:
        >>> tokenize("Сумата е 15 лв.")
        [Token('Сумата', 0, 6), Token('е', 7, 8), Token('15', 9, 11), Token('лв.', 12, 15)]
    """
    tokens: List[Token] = []

    for match in TOKEN_PATTERN.finditer(text):
        # Determine token type from named groups
        raw_type = match.lastgroup or "unknown"
        value = match.group()
        start = match.start()
        end = match.end()

        if raw_type == "whitespace":
            if preserve_whitespace:
                tokens.append(Token(value, start, end, "", "whitespace"))
            elif tokens:
                # Attach whitespace to previous token
                tokens[-1].whitespace_after = value
        else:
            tokens.append(Token(value, start, end, "", raw_type))

    return tokens


def iter_tokens(text: str, preserve_whitespace: bool = False) -> Iterator[Token]:
    """
    Лениво итериране на токени за големи текстове.
    """
    pending_whitespace = ""

    for match in TOKEN_PATTERN.finditer(text):
        raw_type = match.lastgroup or "unknown"
        value = match.group()
        start = match.start()
        end = match.end()

        if raw_type == "whitespace":
            if preserve_whitespace:
                yield Token(value, start, end, "", "whitespace")
            else:
                pending_whitespace = value
        else:
            token = Token(value, start, end, "", raw_type)
            if pending_whitespace:
                # This will be attached to previous token if we had state
                pending_whitespace = ""
            yield token


def reconstruct(tokens: List[Token]) -> str:
    """
    Възстановява оригиналния текст от токени.

    Args:
        tokens: Списък от токени

    Returns:
        Възстановен текст
    """
    if not tokens:
        return ""

    parts = []
    for token in tokens:
        parts.append(token.value)
        if token.whitespace_after:
            parts.append(token.whitespace_after)

    return "".join(parts)


class Tokenizer:
    """Stateful tokenizer with configurable options."""

    def __init__(self, preserve_whitespace: bool = False):
        self.preserve_whitespace = preserve_whitespace

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize text into tokens."""
        return tokenize(text, self.preserve_whitespace)

    def iter_tokens(self, text: str) -> Iterator[Token]:
        """Lazy token iteration for large texts."""
        return iter_tokens(text, self.preserve_whitespace)

    @staticmethod
    def reconstruct(tokens: List[Token]) -> str:
        """Rebuild original text from tokens."""
        return reconstruct(tokens)

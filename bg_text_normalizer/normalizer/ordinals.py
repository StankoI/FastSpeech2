# -*- coding: utf-8 -*-
"""Ordinal number normalization for Bulgarian text."""

from typing import Dict
from ..numbers_to_words import number_to_ordinal


def normalize_ordinal(number: int, gender: str = 'm', with_prefix: bool = False) -> str:
    """
    Нормализира редно числително за произнасяне.

    Args:
        number: Числото
        gender: Род ('m', 'f', 'n')
        with_prefix: Дали да добави "номер" отпред (за №)

    Returns:
        Текст за произнасяне

    Examples:
        >>> normalize_ordinal(15, 'm')
        'петнадесети'
        >>> normalize_ordinal(15, 'm', with_prefix=True)
        'номер петнадесет'
        >>> normalize_ordinal(1, 'f')
        'първа'
    """
    if with_prefix:
        # For № prefix, use cardinal number with "номер"
        from ..numbers_to_words import number_to_words
        return f"номер {number_to_words(number)}"

    return number_to_ordinal(number, gender)


def normalize_ordinal_from_dict(metadata: Dict) -> str:
    """Normalize ordinal from metadata dictionary."""
    number = metadata.get('number', 1)
    gender = metadata.get('gender', 'm')

    # Check if it was a № prefix type
    # (We could track this in metadata if needed)
    return normalize_ordinal(number, gender)

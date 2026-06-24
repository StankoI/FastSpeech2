# -*- coding: utf-8 -*-
"""Phone number normalization for Bulgarian text."""

from ..numbers_to_words import UNITS


def normalize_phone(phone: str) -> str:
    """
    Нормализира телефонен номер за произнасяне.

    Произнася всяка цифра поотделно.

    Args:
        phone: Телефонен номер (с или без форматиране)

    Returns:
        Текст за произнасяне

    Examples:
        >>> normalize_phone("0888123456")
        'нула осем осем осем едно две три четири пет шест'
        >>> normalize_phone("+359 888 123 456")
        'плюс три пет девет осем осем осем едно две три четири пет шест'
    """
    parts = []

    for char in phone:
        if char.isdigit():
            parts.append(UNITS[int(char)])
        elif char == '+':
            parts.append('плюс')
        # Skip spaces, dashes, etc.

    return ' '.join(parts)


def normalize_phone_grouped(phone: str, group_size: int = 2) -> str:
    """
    Нормализира телефон с групиране на цифрите.

    Args:
        phone: Телефонен номер
        group_size: Размер на групите (2 = "осемдесет и осем", 3 = "осемстотин осемдесет и осем")

    Returns:
        Текст за произнасяне с групи

    Examples:
        >>> normalize_phone_grouped("0888123456", 2)
        'нула осемдесет и осем осемдесет и едно двадесет и три четиридесет и пет шест'
    """
    from ..numbers_to_words import number_to_words

    # Extract digits only
    digits = ''.join(c for c in phone if c.isdigit())

    parts = []
    i = 0

    # Handle leading zero specially
    if digits.startswith('0'):
        parts.append('нула')
        i = 1

    # Group remaining digits
    remaining = digits[i:]
    while len(remaining) >= group_size:
        group = remaining[:group_size]
        remaining = remaining[group_size:]
        parts.append(number_to_words(int(group)))

    # Handle remaining digits
    for digit in remaining:
        parts.append(UNITS[int(digit)])

    return ' '.join(parts)

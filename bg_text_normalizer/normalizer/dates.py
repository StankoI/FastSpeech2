# -*- coding: utf-8 -*-
"""Date normalization for Bulgarian text."""

from typing import Dict
from ..numbers_to_words import number_to_ordinal, number_to_words

MONTHS: Dict[int, str] = {
    1: 'януари',
    2: 'февруари',
    3: 'март',
    4: 'април',
    5: 'май',
    6: 'юни',
    7: 'юли',
    8: 'август',
    9: 'септември',
    10: 'октомври',
    11: 'ноември',
    12: 'декември',
}


def normalize_date(day: int, month: int, year: int, include_year: bool = True) -> str:
    """
    Нормализира дата за произнасяне.

    Args:
        day: Ден (1-31)
        month: Месец (1-12)
        year: Година
        include_year: Дали да включва годината

    Returns:
        Текст за произнасяне

    Examples:
        >>> normalize_date(21, 4, 2026)
        'двадесет и първи април две хиляди двадесет и шеста година'
        >>> normalize_date(1, 1, 2000)
        'първи януари две хиляди година'
    """
    parts = []

    # Day - ordinal, masculine (ден is masculine)
    day_word = number_to_ordinal(day, 'm')
    parts.append(day_word)

    # Month name
    month_name = MONTHS.get(month, str(month))
    parts.append(month_name)

    # Year - ordinal feminine (година is feminine)
    if include_year:
        year_word = number_to_ordinal(year, 'f')
        parts.append(f"{year_word} година")

    return ' '.join(parts)


def normalize_date_from_dict(date_info: Dict[str, int]) -> str:
    """Normalize date from metadata dictionary."""
    day = date_info.get('day', 1)
    month = date_info.get('month', 1)
    year = date_info.get('year', 2000)
    return normalize_date(day, month, year)

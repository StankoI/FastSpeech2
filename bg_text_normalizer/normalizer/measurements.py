# -*- coding: utf-8 -*-
"""Measurement normalization for Bulgarian text."""

from typing import Dict
from ..numbers_to_words import number_to_words


def normalize_measurement(value: float, singular: str, plural: str, gender: str) -> str:
    """
    Нормализира измерване за произнасяне.

    Args:
        value: Числова стойност
        singular: Единствено число на единицата
        plural: Множествено число на единицата
        gender: Род на единицата ('m', 'f', 'n')

    Returns:
        Текст за произнасяне

    Examples:
        >>> normalize_measurement(1, 'килограм', 'килограма', 'm')
        'един килограм'
        >>> normalize_measurement(3, 'килограм', 'килограма', 'm')
        'три килограма'
        >>> normalize_measurement(2.5, 'литър', 'литра', 'm')
        'две цяло и пет литра'
    """
    # Handle decimal values
    if isinstance(value, float) and not value.is_integer():
        whole = int(value)
        decimal = round((value - whole) * 10)

        whole_word = number_to_words(whole) if whole > 0 else ""
        decimal_word = number_to_words(decimal)

        # Adjust gender for цяло (neuter)
        if whole == 1:
            whole_word = "едно"
        elif whole == 2:
            whole_word = "две"

        if whole > 0:
            value_word = f"{whole_word} цяло и {decimal_word}"
        else:
            value_word = f"нула цяло и {decimal_word}"

        unit_word = plural  # Decimals use plural
    else:
        int_value = int(value)
        value_word = number_to_words(int_value)

        # Adjust for gender
        if int_value == 1:
            value_word = {'m': 'един', 'f': 'една', 'n': 'едно'}[gender]
            unit_word = singular
        elif int_value == 2:
            value_word = {'m': 'два', 'f': 'две', 'n': 'две'}[gender]
            unit_word = plural
        else:
            unit_word = plural

    return f"{value_word} {unit_word}"


def normalize_measurement_from_dict(metadata: Dict) -> str:
    """Normalize measurement from metadata dictionary."""
    value = metadata.get('value', 0)
    singular = metadata.get('singular', '')
    plural = metadata.get('plural', '')
    gender = metadata.get('gender', 'm')

    return normalize_measurement(value, singular, plural, gender)

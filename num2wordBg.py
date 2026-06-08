# -*- coding: utf-8 -*-
"""Конвертира числа в български думи."""

UNITS = {
    0: 'нула', 1: 'едно', 2: 'две', 3: 'три', 4: 'четири',
    5: 'пет', 6: 'шест', 7: 'седем', 8: 'осем', 9: 'девет'
}

TEENS = {
    10: 'десет', 11: 'единадесет', 12: 'дванадесет', 13: 'тринадесет',
    14: 'четиринадесет', 15: 'петнадесет', 16: 'шестнадесет',
    17: 'седемнадесет', 18: 'осемнадесет', 19: 'деветнадесет'
}

TENS = {
    2: 'двадесет', 3: 'тридесет', 4: 'четиридесет', 5: 'петдесет',
    6: 'шестдесет', 7: 'седемдесет', 8: 'осемдесет', 9: 'деветдесет'
}

HUNDREDS = {
    1: 'сто', 2: 'двеста', 3: 'триста', 4: 'четиристотин',
    5: 'петстотин', 6: 'шестстотин', 7: 'седемстотин',
    8: 'осемстотин', 9: 'деветстотин'
}

# (singular, plural, gender: 'm'=masculine, 'f'=feminine)
SCALES = [
    (1000, 'хиляда', 'хиляди', 'f'),
    (10**6, 'милион', 'милиона', 'm'),
    (10**9, 'милиард', 'милиарда', 'm'),
    (10**12, 'трилион', 'трилиона', 'm'),
    (10**15, 'квадрилион', 'квадрилиона', 'm'),
    (10**18, 'квинтилион', 'квинтилиона', 'm'),
    (10**21, 'секстилион', 'секстилиона', 'm'),
    (10**24, 'септилион', 'септилиона', 'm'),
    (10**27, 'октилион', 'октилиона', 'm'),
    (10**30, 'нонилион', 'нонилиона', 'm'),
    (10**33, 'децилион', 'децилиона', 'm'),
    (10**36, 'ундецилион', 'ундецилиона', 'm'),
    (10**39, 'дуодецилион', 'дуодецилиона', 'm'),
    (10**42, 'тредецилион', 'тредецилиона', 'm'),
    (10**45, 'кватордецилион', 'кватордецилиона', 'm'),
    (10**48, 'квиндецилион', 'квиндецилиона', 'm'),
    (10**51, 'сексдецилион', 'сексдецилиона', 'm'),
    (10**54, 'септдецилион', 'септдецилиона', 'm'),
    (10**57, 'октодецилион', 'октодецилиона', 'm'),
    (10**60, 'новемдецилион', 'новемдецилиона', 'm'),
    (10**63, 'вигинтилион', 'вигинтилиона', 'm'),
    (10**100, 'гугол', 'гугола', 'm'),
]


def _number_under_1000(n: int, gender: str = 'n') -> str:
    """Конвертира число от 0 до 999. gender: 'm', 'f', 'n' (neutral)"""
    if n == 0:
        return ''

    if n < 10:
        if n == 1:
            return {'m': 'един', 'f': 'една', 'n': 'едно'}[gender]
        if n == 2:
            return {'m': 'два', 'f': 'две', 'n': 'две'}[gender]
        return UNITS[n]

    if n < 20:
        return TEENS[n]

    if n < 100:
        tens, units = divmod(n, 10)
        if units == 0:
            return TENS[tens]
        unit_word = _number_under_1000(units, gender)
        return f"{TENS[tens]} и {unit_word}"

    hundreds, remainder = divmod(n, 100)
    if remainder == 0:
        return HUNDREDS[hundreds]

    connector = ' и ' if remainder < 20 else ' '
    return f"{HUNDREDS[hundreds]}{connector}{_number_under_1000(remainder, gender)}"


def _scale_word(count: int, singular: str, plural: str, gender: str) -> str:
    """Генерира дума за мащаб с правилната форма."""
    if count == 1:
        prefix = 'един' if gender == 'm' else ''
        return f"{prefix} {singular}".strip() if gender == 'm' else singular

    count_word = _number_under_1000(count, gender)
    return f"{count_word} {plural}"


def number_to_words(n: int) -> str:
    """
    Конвертира цяло число в български думи.

    Поддържа произволно големи числа (до гугол и повече).

    Примери:
        >>> number_to_words(0)
        'нула'
        >>> number_to_words(21)
        'двадесет и едно'
        >>> number_to_words(123)
        'сто двадесет и три'
        >>> number_to_words(1000)
        'хиляда'
        >>> number_to_words(10**12)
        'един трилион'
    """
    if not isinstance(n, int):
        raise TypeError(f"Очаква се цяло число, получено {type(n).__name__}")

    if n == 0:
        return UNITS[0]

    if n < 0:
        return f"минус {number_to_words(-n)}"

    parts = []

    # Обработваме от най-големия мащаб надолу
    for scale, singular, plural, gender in reversed(SCALES):
        if n >= scale:
            count, n = divmod(n, scale)
            parts.append(_scale_word(count, singular, plural, gender))

    # Остатък под 1000
    if n:
        parts.append(_number_under_1000(n))

    return ' '.join(parts)


def text_numbers_to_words(text: str) -> str:
    """
    Замества всички числа в текст с български думи.

    Примери:
        >>> text_numbers_to_words("Имам 3 ябълки и 12 круши.")
        'Имам три ябълки и дванадесет круши.'
        >>> text_numbers_to_words("Цената е 1500 лева.")
        'Цената е хиляда и петстотин лева.'
    """
    import re

    def replace_number(match):
        return number_to_words(int(match.group()))

    return re.sub(r'\b\d+\b', replace_number, text)


if __name__ == '__main__':
    # Примери
    test_numbers = [0, 1, 10, 11, 21, 100, 101, 111, 123, 200, 1000, 1001,
                    2000, 2345, 10000, 100000, 1000000, 1234567, -42]

    for num in test_numbers:
        print(f"{num:>12} → {number_to_words(num)}")

    print("\n--- Текст с числа ---")
    sample = "Купих 3 книги за 25 лева и 50 стотинки."
    print(f"Вход: {sample}")
    print(f"Изход: {text_numbers_to_words(sample)}")
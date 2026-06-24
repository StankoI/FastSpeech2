# -*- coding: utf-8 -*-
"""Currency normalization for Bulgarian text."""

from typing import Dict, Tuple
from ..numbers_to_words import number_to_words

# Currency info: code -> (singular, plural, gender, subunit_singular, subunit_plural)
CURRENCIES: Dict[str, Tuple[str, str, str, str, str]] = {
    'BGN': ('лев', 'лева', 'm', 'стотинка', 'стотинки'),
    'EUR': ('евро', 'евро', 'n', 'цент', 'цента'),
    'USD': ('долар', 'долара', 'm', 'цент', 'цента'),
    'GBP': ('паунд', 'паунда', 'm', 'пени', 'пенита'),
}


def normalize_currency(amount: float, currency: str = 'BGN') -> str:
    """
    Нормализира сума във валута за произнасяне.

    Args:
        amount: Сумата (може да е с десетични)
        currency: ISO код на валутата

    Returns:
        Текст за произнасяне

    Examples:
        >>> normalize_currency(1, 'BGN')
        'един лев'
        >>> normalize_currency(15, 'BGN')
        'петнадесет лева'
        >>> normalize_currency(2.50, 'BGN')
        'два лева и петдесет стотинки'
    """
    info = CURRENCIES.get(currency, CURRENCIES['BGN'])
    singular, plural, gender, sub_singular, sub_plural = info

    # Split into main and subunits
    if isinstance(amount, float):
        main = int(amount)
        subunits = round((amount - main) * 100)
    else:
        main = int(amount)
        subunits = 0

    parts = []

    # Main amount
    if main > 0 or (main == 0 and subunits == 0):
        main_word = number_to_words(main)

        # Gender adjustment for 1 and 2
        if main == 1:
            main_word = {'m': 'един', 'f': 'една', 'n': 'едно'}[gender]
        elif main == 2:
            main_word = {'m': 'два', 'f': 'две', 'n': 'две'}[gender]

        currency_word = singular if main == 1 else plural
        parts.append(f"{main_word} {currency_word}")

    # Subunits (stotinki, cents, etc.)
    if subunits > 0:
        sub_word = number_to_words(subunits)

        # Subunits are typically feminine in Bulgarian
        if subunits == 1:
            sub_word = 'една'
        elif subunits == 2:
            sub_word = 'две'

        sub_currency = sub_singular if subunits == 1 else sub_plural
        parts.append(f"{sub_word} {sub_currency}")

    return ' и '.join(parts)

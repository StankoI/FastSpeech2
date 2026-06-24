# -*- coding: utf-8 -*-
"""Detection rules for Bulgarian text context analysis."""

import re
from typing import Dict, Optional

# Currency markers and their canonical forms
CURRENCY_MARKERS: Dict[str, str] = {
    'лв.': 'BGN', 'лв': 'BGN', 'лева': 'BGN', 'лев': 'BGN',
    'евро': 'EUR', 'eur': 'EUR', '€': 'EUR',
    'долар': 'USD', 'долара': 'USD', 'долари': 'USD',
    'usd': 'USD', '$': 'USD',
}

# Measurement units: abbreviation -> (singular, plural, gender)
MEASUREMENT_UNITS: Dict[str, tuple] = {
    # Weight
    'кг': ('килограм', 'килограма', 'm'),
    'г': ('грам', 'грама', 'm'),
    'мг': ('милиграм', 'милиграма', 'm'),
    'т': ('тон', 'тона', 'm'),
    # Length
    'м': ('метър', 'метра', 'm'),
    'см': ('сантиметър', 'сантиметра', 'm'),
    'мм': ('милиметър', 'милиметра', 'm'),
    'км': ('километър', 'километра', 'm'),
    # Volume
    'л': ('литър', 'литра', 'm'),
    'мл': ('милилитър', 'милилитра', 'm'),
    # Area
    'кв.м': ('квадратен метър', 'квадратни метра', 'm'),
    'кв. м': ('квадратен метър', 'квадратни метра', 'm'),
    # Volume (cubic)
    'куб.м': ('кубичен метър', 'кубични метра', 'm'),
    'куб. м': ('кубичен метър', 'кубични метра', 'm'),
}

# Известни съкращения: съкращение -> разширена форма (или None).
# None означава, че разпознаваме съкращението (за да не се бърка с
# мерна единица / край на изречение), но НЕ го разширяваме, защото е
# двусмислено или се чете както е.
ABBREVIATIONS: Dict[str, Optional[str]] = {
    # Адреси
    'ул.': 'улица', 'бул.': 'булевард', 'ж.к.': 'жилищен комплекс',
    'ет.': 'етаж', 'ап.': 'апартамент', 'пл.': 'площад',
    'бл.': 'блок', 'вх.': 'вход',
    # Титли и обръщения
    'проф.': 'професор', 'доц.': 'доцент', 'инж.': 'инженер',
    'д-р': 'доктор', 'г-н': 'господин', 'г-жа': 'госпожа',
    'г-ца': 'госпожица', 'акад.': 'академик', 'арх.': 'архитект',
    'адв.': 'адвокат', 'ген.': 'генерал', 'полк.': 'полковник',
    'кап.': 'капитан', 'зам.': 'заместник',
    # Връзки / препратки
    'тел.': 'телефон', 'стр.': 'страница', 'напр.': 'например',
    'вкл.': 'включително', 'др.': 'други',  # 'и др.' = 'и други' (мн.ч.)
    'факс': None, 'е-мейл': None,
    # Двусмислени — разпознаваме, но НЕ разширяваме
    'г.': None,    # година / господин / грам
    'гр.': None,   # град / грам / градус
    'кв.': None,   # квартал / кв. м
    # Правни форми (четат се както са)
    'ООД': None, 'ЕООД': None, 'АД': None, 'ЕТ': None,
}

# Date pattern validation
DATE_PATTERN = re.compile(r'^(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})$')

# Phone pattern validation (Bulgarian formats)
PHONE_PATTERN = re.compile(
    r'^(?:\+359|0)[\s\-]?\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3,4}$'
)

# Ordinal patterns
ORDINAL_PREFIX_PATTERN = re.compile(r'^[№#]\s*(\d+)$')
ORDINAL_SUFFIX_PATTERN = re.compile(
    r'^(\d+)[\-]?(ви|ва|во|ри|ра|ро|ти|та|то|ми|ма|мо)$',
    re.IGNORECASE
)


def is_currency_marker(text: str) -> bool:
    """Check if text is a currency marker."""
    return text.lower() in CURRENCY_MARKERS


def get_currency_code(marker: str) -> str:
    """Get ISO currency code from marker."""
    return CURRENCY_MARKERS.get(marker.lower(), 'BGN')


def is_measurement_unit(text: str) -> bool:
    """Check if text is a measurement unit."""
    return text.lower() in MEASUREMENT_UNITS


def get_unit_info(unit: str) -> tuple:
    """Get unit info (singular, plural, gender)."""
    return MEASUREMENT_UNITS.get(unit.lower(), (unit, unit, 'm'))


def is_date(text: str) -> bool:
    """Check if text matches date pattern."""
    return bool(DATE_PATTERN.match(text))


def parse_date(text: str) -> Dict[str, int]:
    """Parse date string to components."""
    match = DATE_PATTERN.match(text)
    if not match:
        return {}

    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Handle 2-digit years
    if year < 100:
        year = 2000 + year if year < 50 else 1900 + year

    return {'day': day, 'month': month, 'year': year}


def is_phone(text: str) -> bool:
    """Check if text matches Bulgarian phone pattern."""
    # Remove internal whitespace for matching
    cleaned = re.sub(r'\s', '', text)
    return bool(PHONE_PATTERN.match(cleaned)) or bool(re.match(r'^0\d{9}$', cleaned))


def is_ordinal(text: str) -> bool:
    """Check if text is an ordinal number."""
    return bool(ORDINAL_PREFIX_PATTERN.match(text) or ORDINAL_SUFFIX_PATTERN.match(text))


def parse_ordinal(text: str) -> Dict[str, any]:
    """Parse ordinal to get number and gender."""
    prefix_match = ORDINAL_PREFIX_PATTERN.match(text)
    if prefix_match:
        return {'number': int(prefix_match.group(1)), 'gender': 'm'}

    suffix_match = ORDINAL_SUFFIX_PATTERN.match(text)
    if suffix_match:
        number = int(suffix_match.group(1))
        suffix = suffix_match.group(2).lower()
        # Determine gender from suffix
        if suffix in ('ви', 'ри', 'ти', 'ми'):
            gender = 'm'
        elif suffix in ('ва', 'ра', 'та', 'ма'):
            gender = 'f'
        else:
            gender = 'n'
        return {'number': number, 'gender': gender}

    return {}


def _abbreviation_key(text: str) -> Optional[str]:
    """Връща ключа от ABBREVIATIONS, който отговаря на text (или None).

    Съкращение в началото на изречение може да е с главна буква, затова
    пробваме и оригинала, и lower/upper варианта.
    """
    for candidate in (text, text.lower(), text.upper()):
        if candidate in ABBREVIATIONS:
            return candidate
    return None


def is_abbreviation(text: str) -> bool:
    """Check if text is a known abbreviation."""
    return _abbreviation_key(text) is not None


def get_abbreviation_expansion(text: str) -> Optional[str]:
    """Връща разширената форма на съкращение, или None ако няма такава.

    None означава или непознато съкращение, или познато, но двусмислено
    (което се чете както е) — викащият да остави текста непроменен.
    """
    key = _abbreviation_key(text)
    return ABBREVIATIONS[key] if key is not None else None

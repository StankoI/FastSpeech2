"""Dependency-free reader for long-text MFA TextGrid interval tiers."""

import re
from dataclasses import dataclass
from pathlib import Path


INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*"
    r"xmin = ([\d.eE+-]+)\s*"
    r"xmax = ([\d.eE+-]+)\s*"
    r'text = "(.*?)"',
    re.S,
)


@dataclass(frozen=True)
class Interval:
    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class Tier:
    name: str
    _objects: tuple


def parse_tier(textgrid_text, tier_name):
    match = re.search(
        r'name = "{}".*?intervals: size = \d+\s*'
        r"(.*?)(?:\n    item \[|\Z)".format(re.escape(tier_name)),
        textgrid_text,
        re.S,
    )
    if not match:
        raise ValueError("TextGrid has no {!r} tier".format(tier_name))
    return Tier(
        tier_name,
        tuple(
            Interval(float(start), float(end), label.replace('""', '"'))
            for start, end, label in INTERVAL_RE.findall(match.group(1))
        ),
    )


def read_textgrid(path, tier_names=("words", "phones")):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {name: parse_tier(text, name) for name in tier_names}

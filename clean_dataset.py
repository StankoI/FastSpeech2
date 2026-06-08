# -*- coding: utf-8 -*-
"""Filter and merge the Euthymius Bulgarian corpus into a single clean manifest.

Reads the two LibriVox source folders, applies the agreed filters, and writes a
unified manifest (``id|wav_path|text``) plus a report of everything dropped.

Filters (in order, first match wins for the drop reason):
  * missing/unreadable wav
  * duration outside [MIN_DUR, MAX_DUR] seconds
  * digit left over after number expansion (num2wordBg)
  * foreign / garbled letters (Latin, Hebrew, ...)  -> whole line dropped
  * text empty after normalisation

The kept text is lowercased, number-expanded and stripped to Bulgarian words
only (punctuation removed); this becomes the ``.lab`` content for MFA and the
``raw_text`` shown during training. Resampling to 22.05 kHz is left to the
later prepare_align step. No audio is copied or modified here.
"""

import csv
import os
import re
import wave

from num2wordBg import text_numbers_to_words
from text.bulgarian import normalize_text, foreign_letters

# --- configuration ----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    ("Euthymius/reaper_0806_librivox"),
    ("Euthymius/staroplaninski_legendi_eu_0812_librivox-1"),
]
OUT_DIR = os.path.join(ROOT, "filelists")
OUT_MANIFEST = os.path.join(OUT_DIR, "euthymius_clean.csv")
OUT_DROPPED = os.path.join(OUT_DIR, "euthymius_dropped.csv")

MIN_DUR = 0.5    # seconds; drop clips shorter than this
MAX_DUR = 25.0   # seconds; drop clips longer than this
_DIGIT_RE = re.compile(r"\d")


def wav_duration(path):
    """Return clip length in seconds, or None if the file can't be read."""
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
        if rate <= 0:
            return None
        return frames / float(rate)
    except Exception:
        return None


def clean_text(raw):
    """Return (clean_text, drop_reason). drop_reason is None when the line is OK."""
    expanded = text_numbers_to_words(raw)
    if _DIGIT_RE.search(expanded):
        return None, "leftover_digit"
    if foreign_letters(expanded):
        return None, "foreign"
    clean = normalize_text(expanded)
    if not clean:
        return None, "empty_text"
    return clean, None


def iter_source_rows(rel_dir):
    """Yield (id, wav_relpath, raw_text) for each metadata line in a source."""
    meta_path = os.path.join(ROOT, rel_dir, "metadata.csv")
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            base, text = line.split("|", 1)
            base = base.strip()
            wav_rel = "{}/wavs/{}.wav".format(rel_dir, base)
            yield base, wav_rel, text


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    kept = []
    dropped = []  # (id, reason, detail)
    reasons = {}
    kept_seconds = 0.0

    for rel_dir in SOURCES:
        for base, wav_rel, raw_text in iter_source_rows(rel_dir):
            wav_abs = os.path.join(ROOT, wav_rel.replace("/", os.sep))

            if not os.path.exists(wav_abs):
                dropped.append((base, "missing_wav", ""))
                reasons["missing_wav"] = reasons.get("missing_wav", 0) + 1
                continue

            dur = wav_duration(wav_abs)
            if dur is None:
                dropped.append((base, "bad_wav", ""))
                reasons["bad_wav"] = reasons.get("bad_wav", 0) + 1
                continue
            if dur > MAX_DUR:
                dropped.append((base, "too_long", "%.1fs" % dur))
                reasons["too_long"] = reasons.get("too_long", 0) + 1
                continue
            if dur < MIN_DUR:
                dropped.append((base, "too_short", "%.2fs" % dur))
                reasons["too_short"] = reasons.get("too_short", 0) + 1
                continue

            text, reason = clean_text(raw_text)
            if reason is not None:
                dropped.append((base, reason, raw_text[:60]))
                reasons[reason] = reasons.get(reason, 0) + 1
                continue

            kept.append((base, wav_rel, text))
            kept_seconds += dur

    with open(OUT_MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for row in kept:
            w.writerow(row)

    with open(OUT_DROPPED, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["id", "reason", "detail"])
        for row in dropped:
            w.writerow(row)

    total = len(kept) + len(dropped)
    print("=== Euthymius cleaning report ===")
    print("scanned : {}".format(total))
    print("kept    : {}  ({:.2f} hours)".format(len(kept), kept_seconds / 3600))
    print("dropped : {}".format(len(dropped)))
    for r in sorted(reasons):
        print("   - {:<14}: {}".format(r, reasons[r]))
    print("manifest -> {}".format(os.path.relpath(OUT_MANIFEST, ROOT)))
    print("dropped  -> {}".format(os.path.relpath(OUT_DROPPED, ROOT)))


if __name__ == "__main__":
    main()

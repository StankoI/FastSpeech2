#!/usr/bin/env python3
"""Report observed TextGrid phones without modifying the checkpoint inventory."""

from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text.bulgarian_mfa_phones import PHONES


TEXTGRID_ROOT = Path("preprocessed_data/Bulgarian/TextGrid")


def main():
    import tgt

    observed = set()
    files = 0
    blank_intervals = 0
    for path in sorted(TEXTGRID_ROOT.rglob("*.TextGrid")):
        files += 1
        tier = tgt.io.read_textgrid(str(path)).get_tier_by_name("phones")
        for interval in tier._objects:
            phone = unicodedata.normalize("NFC", interval.text.strip())
            if not phone:
                blank_intervals += 1
                phone = "sp"
            elif phone == "sil":
                phone = "sp"
            observed.add(phone)
    if not files:
        raise SystemExit("No TextGrids found under {}".format(TEXTGRID_ROOT))

    unknown = sorted(observed - set(PHONES))
    print("TextGrids:", files)
    print("Observed phones:", len(observed), sorted(observed))
    print("Blank intervals canonicalized to sp:", blank_intervals)
    if unknown:
        raise SystemExit("Phones missing from frozen inventory: {}".format(unknown))
    print("[PASS] observed phones are covered by frozen inventory")


if __name__ == "__main__":
    main()

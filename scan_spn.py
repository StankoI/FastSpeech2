"""Scan MFA TextGrids for `spn` (OOV->spoken-noise) corruption.

Usage:
    python scan_spn.py <path>            # <path> = a .zip of TextGrids OR a dir
    python scan_spn.py <path> --oov oov_words.txt

Reports: # files with >=1 spn, total spn intervals, # utterances total,
% utterances affected, distinct OOV words (word-tier text under each spn),
and how many spn intervals map to MFA's unrecoverable "<unk>" token.
Read-only except for writing the OOV word list. Works on the zip directly
(no extraction needed) or on an extracted directory.
"""
import sys, os, re, zipfile, argparse, io
from collections import Counter

INTERVAL_RE = re.compile(
    r'intervals \[\d+\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "(.*?)"',
    re.S,
)


def parse_tier(text, tiername):
    """Return [(xmin, xmax, label), ...] for the named IntervalTier."""
    m = re.search(
        r'name = "%s".*?intervals: size = \d+\s*(.*?)(?:\n    item \[|\Z)' % tiername,
        text, re.S,
    )
    if not m:
        return []
    out = []
    for im in INTERVAL_RE.finditer(m.group(1)):
        out.append((float(im.group(1)), float(im.group(2)), im.group(3)))
    return out


def word_at(words, t):
    """Word-tier label whose interval contains time t (midpoint of a phone)."""
    for ws, we, wt in words:
        if ws <= t <= we:
            return wt
    return ""


def iter_textgrids(path):
    """Yield (name, text) for every TextGrid in a zip or directory."""
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.endswith(".TextGrid")]
        print(f"[zip] {path}: {len(zf.namelist())} entries total, "
              f"{len(names)} .TextGrid entries")
        print("[zip] first 5 .TextGrid entries:")
        for n in names[:5]:
            print("   ", n)
        if not names:
            print("!! No .TextGrid entries found -- aborting.")
            sys.exit(1)
        print()
        for n in names:
            yield n, zf.read(n).decode("utf-8", "replace")
    else:
        tgs = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".TextGrid"):
                    tgs.append(os.path.join(root, f))
        print(f"[dir] {path}: {len(tgs)} .TextGrid files")
        for n in tgs[:5]:
            print("   ", n)
        if not tgs:
            print("!! No .TextGrid files found -- aborting.")
            sys.exit(1)
        print()
        for n in tgs:
            with io.open(n, encoding="utf-8", errors="replace") as fh:
                yield n, fh.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="zip of TextGrids or a directory")
    ap.add_argument("--oov", default=None, help="write distinct OOV words here")
    args = ap.parse_args()

    total = 0
    files_with_spn = 0
    total_spn = 0
    unk_spn = 0                      # spn intervals whose word == "<unk>"
    oov_words = Counter()            # real (recoverable) OOV words -> occurrences

    for name, text in iter_textgrids(args.path):
        total += 1
        phones = parse_tier(text, "phones")
        words = parse_tier(text, "words")
        n_spn = 0
        for ps, pe, pt in phones:
            if pt == "spn":
                n_spn += 1
                total_spn += 1
                w = word_at(words, (ps + pe) / 2.0).strip().lower()
                if w == "<unk>" or w == "":
                    unk_spn += 1
                else:
                    oov_words[w] += 1
        if n_spn:
            files_with_spn += 1

    pct_files = 100.0 * files_with_spn / total if total else 0.0
    pct_unk = 100.0 * unk_spn / total_spn if total_spn else 0.0

    print("================ SPN SCAN RESULTS ================")
    print(f"utterances (TextGrids) scanned : {total}")
    print(f"utterances with >=1 spn        : {files_with_spn}  ({pct_files:.2f}%)")
    print(f"total spn intervals            : {total_spn}")
    print(f"  -> recoverable OOV words     : {total_spn - unk_spn}")
    print(f"  -> unrecoverable <unk>       : {unk_spn}  ({pct_unk:.2f}% of spn)")
    print(f"distinct recoverable OOV words : {len(oov_words)}")
    print("top 20 OOV words by frequency  :")
    for w, c in oov_words.most_common(20):
        print(f"    {c:6d}  {w}")
    print("==================================================")

    if args.oov:
        with io.open(args.oov, "w", encoding="utf-8") as f:
            for w, _ in sorted(oov_words.items()):
                f.write(w + "\n")
        print(f"wrote {len(oov_words)} distinct OOV words -> {args.oov}")


if __name__ == "__main__":
    main()

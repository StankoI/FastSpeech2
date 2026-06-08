# -*- coding: utf-8 -*-
"""Generate the MFA grapheme pronunciation dictionary for Bulgarian.

Reads the cleaned manifest, collects every unique word, and writes a dictionary
where each word maps to its sequence of Cyrillic letters (the grapheme tokens
defined in text/bulgarian.py). MFA uses this to align the corpus; because the
"phones" are letters, the MFA acoustic model must be trained from scratch.

Output format (one entry per line, tab-separated, whitespace-separated phones):
    дума<TAB>д у м а
"""

import os

from text.bulgarian import word_to_phonemes

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "filelists", "euthymius_clean.csv")
OUT_DICT = os.path.join(ROOT, "lexicon", "bulgarian-grapheme.txt")


def main():
    words = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            text = line.split("|")[-1]
            for w in text.split():
                words.add(w)

    entries = {}
    skipped = 0
    for w in words:
        phones = word_to_phonemes(w)
        if not phones:
            skipped += 1
            continue
        entries[w] = phones

    os.makedirs(os.path.dirname(OUT_DICT), exist_ok=True)
    with open(OUT_DICT, "w", encoding="utf-8", newline="\n") as f:
        for w in sorted(entries):
            f.write("{}\t{}\n".format(w, " ".join(entries[w])))

    phone_set = sorted({p for ph in entries.values() for p in ph})
    print("=== Bulgarian grapheme dictionary ===")
    print("unique words   : {}".format(len(words)))
    print("entries written: {}".format(len(entries)))
    print("skipped (empty): {}".format(skipped))
    print("phone set ({}) : {}".format(len(phone_set), " ".join(phone_set)))
    print("dictionary -> {}".format(os.path.relpath(OUT_DICT, ROOT)))


if __name__ == "__main__":
    main()

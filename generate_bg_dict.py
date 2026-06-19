# -*- coding: utf-8 -*-
"""Generate the MFA grapheme pronunciation dictionary for Bulgarian.

SINGLE SOURCE OF TRUTH: the dictionary is built from the same manifest the model
trains on (``RealData/merged_dataset/manifest.csv``). Every transcript is passed
through ``text.bulgarian.normalize_text`` (the identical normalization used to
write the MFA ``.lab`` files), every unique word is collected, and each word maps
to its sequence of Cyrillic letters (the grapheme tokens defined in
text/bulgarian.py). Because the dictionary words and the alignment text come from
one normalized source, the corpus has zero OOV by construction.

The "phones" are letters, so the MFA acoustic model must be trained from scratch.

Output format (one entry per line, tab-separated, whitespace-separated phones):
    дума<TAB>д у м а
"""

import os

from text.bulgarian import word_to_phonemes, normalize_text

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "RealData", "merged_dataset", "manifest.csv")
OUT_DICT = os.path.join(ROOT, "lexicon", "dictionary_full.txt")


def main():
    words = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            # manifest is `id|wav_path|text`; the transcript is the last field
            text = line.split("|")[-1]
            for w in normalize_text(text).split():
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
    print("=== Bulgarian grapheme dictionary (from RealData manifest) ===")
    print("manifest       : {}".format(os.path.relpath(MANIFEST, ROOT)))
    print("unique words   : {}".format(len(words)))
    print("entries written: {}".format(len(entries)))
    print("skipped (empty): {}".format(skipped))
    print("phone set ({}) : {}".format(len(phone_set), " ".join(phone_set)))
    print("dictionary -> {}".format(os.path.relpath(OUT_DICT, ROOT)))


if __name__ == "__main__":
    main()

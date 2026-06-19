# -*- coding: utf-8 -*-
"""Build the MFA alignment corpus from the SINGLE SOURCE OF TRUTH manifest.

For every row of ``RealData/merged_dataset/manifest.csv`` this writes a
``{uid}.lab`` whose text is ``normalize_text(transcript)`` -- the IDENTICAL
normalization used to build ``lexicon/dictionary_full.txt`` -- next to a hardlink
to the matching wav, grouped into pseudo-speakers (``book__pista``) so MFA runs
multi-threaded. Because the .lab words and the dictionary words come from the same
normalized manifest, the corpus has zero OOV by construction; this is verified at
the end against the dictionary.

Usage:
    python build_mfa_corpus.py                 # -> mfa_corpus_real/
    python build_mfa_corpus.py <out_dir> <dict>
"""
import os, sys, importlib.util

MANIFEST = "RealData/merged_dataset/manifest.csv"
WAVS = "RealData/merged_dataset/wavs"
OUT = sys.argv[1] if len(sys.argv) > 1 else "mfa_corpus_real"
DICT = sys.argv[2] if len(sys.argv) > 2 else "lexicon/dictionary_full.txt"

# load normalize_text in isolation (no package __init__ side effects)
spec = importlib.util.spec_from_file_location("bg", "text/bulgarian.py")
bg = importlib.util.module_from_spec(spec); spec.loader.exec_module(bg)


def main():
    os.makedirs(OUT, exist_ok=True)
    n_lab = n_wav = n_missing = 0
    speakers = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("|")
            if len(a) < 3:
                continue
            uid, _wavrel, text = a[0], a[1], a[2]
            spk = "__".join(uid.split("__")[:2]) or "spk0"
            speakers.add(spk)
            sdir = os.path.join(OUT, spk)
            os.makedirs(sdir, exist_ok=True)

            # .lab text = normalize_text(transcript) -- same fn as the dictionary
            with open(os.path.join(sdir, uid + ".lab"), "w",
                      encoding="utf-8", newline="\n") as lf:
                lf.write(bg.normalize_text(text))
            n_lab += 1

            src = os.path.join(WAVS, uid + ".wav")
            dst = os.path.join(sdir, uid + ".wav")
            if not os.path.exists(src):
                n_missing += 1
                continue
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)          # hardlink (same volume, no copy)
                except OSError:
                    import shutil; shutil.copy(src, dst)
            n_wav += 1

    print("=== MFA corpus built from manifest ===")
    print("out dir        :", OUT)
    print("pseudo-speakers:", len(speakers))
    print(".lab written   :", n_lab)
    print("wav linked     :", n_wav)
    print("wav missing    :", n_missing)

    # ---- zero-OOV verification against the dictionary ----
    dict_words = set()
    with open(DICT, encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                dict_words.add(line.split("\t", 1)[0])
    oov = set()
    scanned = 0
    for root, _, files in os.walk(OUT):
        for fn in files:
            if fn.endswith(".lab"):
                scanned += 1
                with open(os.path.join(root, fn), encoding="utf-8") as lf:
                    for w in lf.read().split():
                        if w not in dict_words:
                            oov.add(w)
    print("--- consistency check ---")
    print("dictionary words   :", len(dict_words))
    print(".lab files scanned :", scanned)
    print("OOV words in corpus:", len(oov), "<-- must be 0")
    if oov:
        print("  examples:", list(sorted(oov))[:20])
    else:
        print("PASS: every word in every .lab is in dictionary_full.txt (0 OOV).")


if __name__ == "__main__":
    main()

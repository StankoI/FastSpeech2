"""Character-frequency analysis of a FastSpeech2 manifest (id|wav_path|text).

Usage:  python analyze_chars.py [path_to_manifest]
Default path: filelists/train_subset.csv
Reports per-character counts, per-clip coverage, and the дж/дз affricate digraphs,
so we can see which graphemes (e.g. ю, дж) are under-represented in the training split.
"""
import sys, collections, unicodedata

PATH = sys.argv[1] if len(sys.argv) > 1 else "filelists/train_subset.csv"

texts = []
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            texts.append(parts[2])
        elif len(parts) == 1:           # fallback: plain text file, one line per clip
            texts.append(parts[0])

n_clips = len(texts)
char_counts = collections.Counter(c for t in texts for c in t if not c.isspace())
char_clips = collections.Counter()
for t in texts:
    for c in set(t):
        if not c.isspace():
            char_clips[c] += 1
total = sum(char_counts.values())

print(f"file: {PATH}")
print(f"clips: {n_clips:,} | non-space chars: {total:,} | unique chars: {len(char_counts)}\n")
print(f"{'char':>5} {'count':>10} {'%':>7} {'#clips':>8} {'%clips':>7}  name")
for c, n in char_counts.most_common():
    name = unicodedata.name(c, "?")
    print(f"{c!r:>5} {n:>10,} {100*n/total:>6.2f}% {char_clips[c]:>8,} {100*char_clips[c]/n_clips:>6.1f}%  {name}")

print("\n--- affricate digraphs (model must learn to blend these) ---")
joined = "".join(texts)
for dg in ["дж", "дз"]:
    occ = joined.count(dg)
    clips = sum(1 for t in texts if dg in t)
    print(f"  {dg!r}: {occ:,} occurrences in {clips:,} clips ({100*clips/max(n_clips,1):.1f}% of clips)")

print("\n--- flagged: graphemes in <5% of clips (likely under-learned) ---")
rare = [(c, n) for c, n in char_counts.items() if char_clips[c] < 0.05 * n_clips]
for c, n in sorted(rare, key=lambda x: x[1]):
    print(f"  {c!r}: {n:,} times, in {char_clips[c]:,} clips ({100*char_clips[c]/n_clips:.1f}%)  {unicodedata.name(c,'?')}")
if not rare:
    print("  (none — every grapheme appears in >5% of clips)")

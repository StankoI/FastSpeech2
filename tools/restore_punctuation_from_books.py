#!/usr/bin/env python3
"""Restore real book punctuation onto the existing clip manifest.

Matching is chronological within each book and uses the exact word-only clip
transcript.  A narrowly constrained fuzzy fallback handles split/merged or
foreign-letter normalization defects, records every such ID, and still requires
the recovered text to normalize back to the existing MFA transcript.
"""

import argparse
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bulgarian_normalization import (
    normalize_for_mfa,
    prosody_words,
    render_prosody_words,
)


def numeric(value):
    match = re.search(r"\d+", value)
    return int(match.group()) if match else -1


def load_dataset(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"uid", "book", "pista", "clip_id", "text"}
    if not rows or not required <= set(rows[0]):
        raise SystemExit("{} must contain columns {}".format(path, sorted(required)))
    if len({row["uid"] for row in rows}) != len(rows):
        raise SystemExit("Duplicate uid in {}".format(path))
    return rows


def find_exact(haystack, needle, cursor):
    padded_needle = " " + needle + " "
    position = haystack.find(padded_needle, cursor)
    return position, padded_needle


def transfer_fuzzy_punctuation(source_pairs, target_words):
    source_words = [word for word, _ in source_pairs]
    ratio = difflib.SequenceMatcher(
        None, "".join(source_words), "".join(target_words), autojunk=False
    ).ratio()
    if ratio < 0.85 or abs(len(source_words) - len(target_words)) > 3:
        return None, ratio

    target_tokens = [None] * len(target_words)
    matcher = difflib.SequenceMatcher(None, source_words, target_words, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(a1 - a0):
                target_tokens[b0 + offset] = source_pairs[a0 + offset][1]
            continue
        source_marks = [token for _, token in source_pairs[a0:a1] if token]
        if len(source_marks) > 1 or b0 == b1:
            return None, ratio
        if source_marks:
            target_tokens[b1 - 1] = source_marks[0]
    return list(zip(target_words, target_tokens)), ratio


def recover_book(rows, book_path, allow_fuzzy):
    source_pairs = prosody_words(book_path.read_text(encoding="utf-8"))
    source_words = [word for word, _ in source_pairs]
    normalized = " ".join(source_words)
    haystack = " " + normalized + " "
    starts = {}
    char_position = 0
    for index, word in enumerate(source_words):
        starts[char_position] = index
        char_position += len(word) + 1

    recovered = {}
    fuzzy = []
    cursor_char = 0
    cursor_word = 0
    skipped_words = 0
    for row_index, row in enumerate(rows):
        target = normalize_for_mfa(row["text"])
        target_words = target.split()
        position, padded_needle = find_exact(haystack, target, cursor_char)
        if position >= 0:
            start_word = starts[position]
            skipped_words += start_word - cursor_word
            end_word = start_word + len(target_words)
            pairs = source_pairs[start_word:end_word]
            cursor_word = end_word
            cursor_char = position + len(padded_needle) - 1
        else:
            if not allow_fuzzy or row_index + 1 >= len(rows):
                raise RuntimeError("No chronological source match for {}: {!r}".format(row["uid"], target))
            next_target = normalize_for_mfa(rows[row_index + 1]["text"])
            next_position, _ = find_exact(haystack, next_target, cursor_char)
            if next_position < 0 or next_position not in starts:
                raise RuntimeError("Cannot bound fuzzy source span for {}".format(row["uid"]))
            next_word = starts[next_position]
            candidate = source_pairs[cursor_word:next_word]
            pairs, ratio = transfer_fuzzy_punctuation(candidate, target_words)
            if pairs is None:
                raise RuntimeError(
                    "Unsafe fuzzy source match for {} (ratio {:.3f}): {!r} vs {!r}".format(
                        row["uid"], ratio, [x[0] for x in candidate], target_words
                    )
                )
            fuzzy.append(
                {
                    "uid": row["uid"],
                    "ratio": ratio,
                    "source_words": [word for word, _ in candidate],
                    "target_words": target_words,
                }
            )
            cursor_word = next_word
            cursor_char = next_position

        punctuated = render_prosody_words(pairs)
        if normalize_for_mfa(punctuated) != target:
            raise RuntimeError("Recovered text changes MFA words for {}".format(row["uid"]))
        recovered[row["uid"]] = punctuated

    return recovered, fuzzy, skipped_words, len(source_words) - cursor_word


def read_pipe_manifest(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                raise SystemExit("{}:{} must be id|wav_path|text".format(path, line_number))
            rows.append(parts)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", default=str(Path.home() / "TTS/data/full_books"))
    parser.add_argument("--dataset", default="bg_realdata/dataset_full.csv")
    parser.add_argument("--dropped", default="bg_realdata/dropped.csv")
    parser.add_argument("--manifest", default="bg_realdata/manifest.csv")
    parser.add_argument("--output", default="bg_realdata/manifest_punctuated.csv")
    parser.add_argument("--report", default="bg_realdata/punctuation_restore_report.json")
    parser.add_argument("--allow-fuzzy", action="store_true")
    args = parser.parse_args()

    books = {path.stem: path for path in Path(args.books).glob("*.txt")}
    dataset_rows = load_dataset(Path(args.dataset))
    dropped_rows = load_dataset(Path(args.dropped)) if args.dropped else []
    duplicate_ids = {row["uid"] for row in dataset_rows} & {
        row["uid"] for row in dropped_rows
    }
    if duplicate_ids:
        raise SystemExit("IDs occur in both dataset and dropped CSV: {}".format(sorted(duplicate_ids)[:20]))
    matching_rows = dataset_rows + dropped_rows
    grouped = defaultdict(list)
    for row in matching_rows:
        grouped[row["book"]].append(row)
    if set(grouped) != set(books):
        raise SystemExit(
            "Book set mismatch: dataset_only={} files_only={}".format(
                sorted(set(grouped) - set(books)), sorted(set(books) - set(grouped))
            )
        )

    recovered = {}
    fuzzy = []
    book_stats = {}
    for book, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (numeric(row["pista"]), numeric(row["clip_id"])))
        mapping, book_fuzzy, skipped, trailing = recover_book(
            rows, books[book], args.allow_fuzzy
        )
        recovered.update(mapping)
        fuzzy.extend(book_fuzzy)
        book_stats[book] = {
            "clips": len(rows),
            "exact": len(rows) - len(book_fuzzy),
            "fuzzy": len(book_fuzzy),
            "skipped_source_words": skipped,
            "trailing_source_words": trailing,
        }

    manifest_rows = read_pipe_manifest(Path(args.manifest))
    manifest_ids = {uid for uid, _, _ in manifest_rows}
    missing = manifest_ids - set(recovered)
    if missing:
        raise SystemExit("No recovered punctuation for manifest IDs: {}".format(sorted(missing)[:20]))

    punctuation = Counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        for uid, wav_path, old_text in manifest_rows:
            text = recovered[uid]
            if normalize_for_mfa(text) != normalize_for_mfa(old_text):
                raise RuntimeError("Output changes existing words for {}".format(uid))
            punctuation.update(
                token
                for _, token in prosody_words(text, expand_numbers=False)
                if token
            )
            handle.write("{}|{}|{}\n".format(uid, wav_path, text))

    report = {
        "books": book_stats,
        "dataset_rows": len(dataset_rows),
        "dropped_anchor_rows": len(dropped_rows),
        "matching_rows": len(matching_rows),
        "manifest_rows": len(manifest_rows),
        "fuzzy_matches": fuzzy,
        "punctuation_counts": dict(punctuation),
        "punctuated_manifest_rows": sum(
            any(
                token
                for _, token in prosody_words(
                    recovered[uid], expand_numbers=False
                )
            )
            for uid in manifest_ids
        ),
        "output": str(output),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("[PASS] restored manifest:", output)
    print("Exact matches:", len(matching_rows) - len(fuzzy))
    print("Fuzzy matches:", len(fuzzy))
    print("Punctuation:", dict(punctuation))
    print("Report:", report_path)


if __name__ == "__main__":
    main()

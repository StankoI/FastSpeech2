#!/usr/bin/env python3
"""Build the punctuation sidecar without rewriting WAVs or rerunning MFA."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bulgarian_normalization import (
    NORMALIZER_VERSION,
    normalize_for_mfa,
    normalize_with_punctuation,
    prosody_words,
)


def _read_manifest(path):
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                raise ValueError("{}:{} must be id|wav_path|text".format(path, line_number))
            uid, wav_path, text = parts
            if not uid or uid in seen:
                raise ValueError("Empty or duplicate manifest id: {}".format(uid))
            seen.add(uid)
            yield uid, wav_path, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/Bulgarian/preprocess.yaml")
    parser.add_argument(
        "--manifest",
        help="punctuated id|wav_path|text manifest; defaults to config path",
    )
    parser.add_argument(
        "--allow-unpunctuated",
        action="store_true",
        help="build a wb-only diagnostic sidecar even if strict coverage fails",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    source = Path(args.manifest or config["path"]["manifest_path"])
    raw = Path(config["path"]["raw_path"]) / "Bulgarian"
    destination = Path(config["path"]["prosody_manifest_path"])
    entries = {}
    punctuated = 0
    for uid, _, source_text in _read_manifest(source):
        lab = raw / (uid + ".lab")
        if not lab.is_file():
            continue  # deliberately rejected during prepare_align
        mfa_text = normalize_for_mfa(source_text)
        actual = lab.read_text(encoding="utf-8").strip()
        if actual != mfa_text:
            raise SystemExit(
                "{} does not match source manifest after normalization: {!r} != {!r}".format(
                    uid, actual, mfa_text
                )
            )
        has_punctuation = any(token for _, token in prosody_words(source_text))
        punctuated += int(has_punctuation)
        entries[uid] = {
            "text": normalize_with_punctuation(source_text),
            "mfa_text": mfa_text,
            "has_punctuation": has_punctuation,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "normalizer_version": NORMALIZER_VERSION,
                "source_manifest": str(source),
                "entries": entries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print("[PASS] prosody entries:", len(entries))
    print("Entries containing source punctuation:", punctuated)
    if not punctuated:
        print("WARNING: source manifest contains no punctuation; exact punctuation cannot be recovered.")
    print("Wrote:", destination)
    if not args.allow_unpunctuated:
        ratio = punctuated / len(entries) if entries else 0
        minimum = float(config.get("prosody", {}).get("min_punctuated_utterance_ratio", 0))
        counts = {}
        for entry in entries.values():
            for _, token in prosody_words(entry["text"], expand_numbers=False):
                if token:
                    counts[token] = counts.get(token, 0) + 1
        missing = [
            token
            for token in config.get("prosody", {}).get("required_tokens", [])
            if counts.get(token, 0) == 0
        ]
        if ratio < minimum or missing:
            raise SystemExit(
                "Prosody gate failed: punctuated_ratio={:.2%}, minimum={:.2%}, "
                "missing_required={}. Pass a genuinely punctuated --manifest; "
                "do not infer punctuation from stripped text.".format(
                    ratio, minimum, missing
                )
            )


if __name__ == "__main__":
    main()

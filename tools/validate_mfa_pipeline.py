#!/usr/bin/env python3
"""Strict validation gates for Bulgarian MFA alignment and FS2 preprocessing."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text import text_to_sequence
from text.bulgarian_mfa_phones import PHONES


INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*"
    r"xmin = ([\d.eE+-]+)\s*"
    r"xmax = ([\d.eE+-]+)\s*"
    r'text = "(.*?)"',
    re.S,
)


def parse_tier(textgrid_text, tier_name):
    match = re.search(
        r'name = "{}".*?intervals: size = \d+\s*'
        r"(.*?)(?:\n    item \[|\Z)".format(re.escape(tier_name)),
        textgrid_text,
        re.S,
    )
    if not match:
        raise ValueError("TextGrid has no {!r} tier".format(tier_name))
    return [
        (float(start), float(end), label)
        for start, end, label in INTERVAL_RE.findall(match.group(1))
    ]


def interval_word(words, midpoint):
    for start, end, label in words:
        if start <= midpoint <= end:
            return label.strip().lower()
    return ""


def validate_alignment(config, allow_spn_words):
    raw_root = Path(config["path"]["raw_path"])
    preprocessed = Path(config["path"]["preprocessed_path"])
    grid_root = preprocessed / "TextGrid"
    run_metadata_path = grid_root / "alignment_run.json"
    if not run_metadata_path.is_file():
        raise SystemExit("Missing alignment metadata: {}".format(run_metadata_path))
    with open(run_metadata_path, encoding="utf-8") as handle:
        run_metadata = json.load(handle)
    failed_ids = set(run_metadata.get("failed_ids", []))

    wavs = {(p.parent.name, p.stem) for p in raw_root.rglob("*.wav")}
    labs = {(p.parent.name, p.stem) for p in raw_root.rglob("*.lab")}
    grids = {(p.parent.name, p.stem) for p in grid_root.rglob("*.TextGrid")}
    missing_grids = wavs - grids
    missing_grid_ids = {basename for _, basename in missing_grids}
    if wavs != labs or grids - wavs or missing_grid_ids != failed_ids:
        raise SystemExit(
            "Coverage mismatch: wavs={} labs={} grids={} missing_labs={} "
            "missing_grids={} extra_grids={}".format(
                len(wavs),
                len(labs),
                len(grids),
                sorted(wavs - labs)[:20],
                sorted(missing_grids)[:20],
                sorted(grids - wavs)[:20],
            )
        )

    expected_metadata = {
        "status": "complete",
        "utterance_count": len(wavs),
        "textgrid_count": len(grids),
        "failed_count": len(failed_ids),
    }
    metadata_mismatches = {
        key: {"metadata": run_metadata.get(key), "actual": value}
        for key, value in expected_metadata.items()
        if run_metadata.get(key) != value
    }
    actual_failed_ratio = len(failed_ids) / len(wavs) if wavs else 1.0
    recorded_ratio = run_metadata.get("failed_ratio")
    if recorded_ratio is None or abs(recorded_ratio - actual_failed_ratio) > 1e-12:
        metadata_mismatches["failed_ratio"] = {
            "metadata": recorded_ratio,
            "actual": actual_failed_ratio,
        }
    maximum_failed_ratio = config.get("mfa", {}).get("max_failed_ratio", 0.01)
    if actual_failed_ratio > maximum_failed_ratio:
        metadata_mismatches["max_failed_ratio"] = {
            "actual": actual_failed_ratio,
            "maximum": maximum_failed_ratio,
        }
    if metadata_mismatches:
        raise SystemExit(
            "Alignment metadata mismatch: {}".format(metadata_mismatches)
        )

    dictionary_path = Path(
        config.get("mfa", {}).get("alignment_dictionary")
        or config["path"]["runtime_lexicon_path"]
    )
    if not dictionary_path.is_file():
        raise SystemExit("Missing alignment/runtime dictionary: {}".format(dictionary_path))
    dictionary_sha256 = hashlib.sha256(dictionary_path.read_bytes()).hexdigest()
    recorded_dictionary_sha256 = run_metadata.get("dictionary", {}).get("sha256")
    if dictionary_sha256 != recorded_dictionary_sha256:
        raise SystemExit(
            "Runtime dictionary differs from alignment snapshot: actual={} "
            "alignment={}".format(dictionary_sha256, recorded_dictionary_sha256)
        )

    allowed = set(PHONES)
    observed = Counter()
    spn_words = Counter()
    blank_intervals = 0
    files_with_internal_pause = 0
    for path in sorted(grid_root.rglob("*.TextGrid")):
        textgrid_text = path.read_text(encoding="utf-8", errors="replace")
        phones = parse_tier(textgrid_text, "phones")
        words = parse_tier(textgrid_text, "words")
        internal_pause = False
        for index, (start, end, label) in enumerate(phones):
            phone = unicodedata.normalize("NFC", label.strip())
            if not phone:
                blank_intervals += 1
                phone = "sp"
                if index not in {0, len(phones) - 1}:
                    internal_pause = True
            elif phone == "sil":
                phone = "sp"
            observed[phone] += 1
            if phone == "spn":
                word = interval_word(words, (start + end) / 2)
                if word and word not in {"<unk>", "[noise]", "[laughter]"}:
                    spn_words[word] += 1
        files_with_internal_pause += int(internal_pause)

    unknown = sorted(set(observed) - allowed)
    if unknown:
        raise SystemExit("Unknown phones outside frozen inventory: {}".format(unknown))
    total_spn_words = sum(spn_words.values())
    if total_spn_words > allow_spn_words:
        raise SystemExit(
            "Found {} lexical words aligned as spn (limit {}). Top: {}".format(
                total_spn_words, allow_spn_words, spn_words.most_common(20)
            )
        )

    print("[PASS] alignment coverage:", len(wavs))
    print("[PASS] alignment metadata and runtime dictionary match")
    print("Explicit MFA export failures excluded:", len(failed_ids))
    print("[PASS] frozen inventory covers observed phones:", len(observed))
    print("Blank MFA intervals mapped to sp:", blank_intervals)
    print("Files with internal pause:", files_with_internal_pause)
    print("Lexical spn intervals:", total_spn_words)


def metadata_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("|", 3)
            if len(parts) != 4:
                raise ValueError("{}:{} malformed metadata".format(path, line_number))
            yield parts


def validate_preprocessed(config):
    root = Path(config["path"]["preprocessed_path"])
    checked = 0
    metadata_ids = set()
    for split in ("train.txt", "val.txt"):
        path = root / split
        if not path.is_file():
            raise SystemExit("Missing {}".format(path))
        for basename, speaker, phone_text, _ in metadata_rows(path):
            key = (speaker, basename)
            if key in metadata_ids:
                raise SystemExit("Duplicate preprocessed metadata ID: {}".format(key))
            metadata_ids.add(key)
            ids = text_to_sequence(
                phone_text,
                config["preprocessing"]["text"]["text_cleaners"],
            )
            duration = np.load(root / "duration" / "{}-duration-{}.npy".format(speaker, basename))
            pitch = np.load(root / "pitch" / "{}-pitch-{}.npy".format(speaker, basename))
            energy = np.load(root / "energy" / "{}-energy-{}.npy".format(speaker, basename))
            mel = np.load(root / "mel" / "{}-mel-{}.npy".format(speaker, basename))

            lengths = {
                "phones": len(ids),
                "duration": len(duration),
                "pitch": len(pitch),
                "energy": len(energy),
            }
            if len(set(lengths.values())) != 1:
                raise SystemExit("{} length mismatch: {}".format(basename, lengths))
            if int(np.sum(duration)) != int(mel.shape[0]):
                raise SystemExit(
                    "{} duration sum {} != mel frames {}".format(
                        basename, int(np.sum(duration)), mel.shape[0]
                    )
                )
            for name, values in (("pitch", pitch), ("energy", energy), ("mel", mel)):
                if not np.all(np.isfinite(values)):
                    raise SystemExit("{} contains non-finite {}".format(basename, name))
            checked += 1

    for required in ("speakers.json", "stats.json"):
        with open(root / required, encoding="utf-8") as handle:
            json.load(handle)
    feature_report_path = root / "feature_extraction_report.json"
    if not feature_report_path.is_file():
        raise SystemExit("Missing {}".format(feature_report_path))
    with feature_report_path.open(encoding="utf-8") as handle:
        feature_report = json.load(handle)
    grids = list((root / "TextGrid").rglob("*.TextGrid"))
    report_expected = {
        "input_count": len(grids),
        "accepted_count": checked,
        "rejected_count": len(grids) - checked,
    }
    report_mismatch = {
        key: {"report": feature_report.get(key), "actual": value}
        for key, value in report_expected.items()
        if feature_report.get(key) != value
    }
    if report_mismatch:
        raise SystemExit("Feature report mismatch: {}".format(report_mismatch))
    rejected_entries = feature_report.get("rejected", [])
    if len(rejected_entries) != feature_report["rejected_count"]:
        raise SystemExit("Feature report rejected list/count mismatch")
    rejected_ratio = (
        feature_report["rejected_count"] / feature_report["input_count"]
        if feature_report["input_count"]
        else 1.0
    )
    if abs(feature_report.get("rejected_ratio", -1) - rejected_ratio) > 1e-12:
        raise SystemExit("Feature report rejected ratio mismatch")
    maximum_ratio = config["preprocessing"].get("max_feature_rejected_ratio", 0.01)
    if rejected_ratio > maximum_ratio:
        raise SystemExit(
            "Feature rejection ratio {:.2%} exceeds {:.2%}".format(
                rejected_ratio, maximum_ratio
            )
        )
    print("[PASS] preprocessed utterances:", checked)
    print("Feature extraction rejections:", feature_report["rejected_count"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/Bulgarian/preprocess.yaml")
    parser.add_argument(
        "--stage", choices=["alignment", "preprocessed", "all"], default="all"
    )
    parser.add_argument(
        "--allow-spn-words",
        type=int,
        default=0,
        help="Explicit tolerance for lexical words aligned as spoken noise",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.stage in {"alignment", "all"}:
        validate_alignment(config, args.allow_spn_words)
    if args.stage in {"preprocessed", "all"}:
        validate_preprocessed(config)


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from bulgarian_normalization import (
    NORMALIZER_VERSION,
    normalize_for_mfa,
    normalize_with_punctuation,
    prosody_words,
)


def _read_manifest(path):
    seen = set()
    with open(path, encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, 1):
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                raise ValueError(
                    "{}:{} must be id|wav_path|text".format(path, line_number)
                )
            basename, wav_path, raw_text = parts
            if not basename:
                raise ValueError("{}:{} has an empty id".format(path, line_number))
            if basename in seen:
                raise ValueError("Duplicate manifest id: {}".format(basename))
            seen.add(basename)
            yield basename, wav_path, raw_text


def _resolve_wav_path(wav_path, base, manifest_parent):
    """Resolve absolute, corpus-relative, or manifest-relative wav paths."""
    path = Path(wav_path)
    if path.is_absolute():
        return path
    candidates = [base / path, manifest_parent / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Return the configured interpretation for an actionable error report.
    return candidates[0]


def prepare_align(config):
    """Create the single-speaker wav/lab tree consumed by MFA and preprocessing.

    A machine-readable report is written next to the output so later stages can
    distinguish deliberately rejected rows from missing alignment output.
    """
    base = Path(config["path"]["corpus_path"])
    manifest = Path(config["path"]["manifest_path"])
    out_dir = Path(config["path"]["raw_path"])
    speaker = "Bulgarian"
    speaker_dir = out_dir / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    rows = list(_read_manifest(manifest))
    punctuation_counts = {}
    punctuated_rows = 0
    for _, _, source_text in rows:
        tokens = [token for _, token in prosody_words(source_text) if token]
        punctuated_rows += int(bool(tokens))
        for token in tokens:
            punctuation_counts[token] = punctuation_counts.get(token, 0) + 1
    prosody_config = config.get("prosody", {})
    punctuated_ratio = punctuated_rows / len(rows) if rows else 0
    minimum_punctuated = float(
        prosody_config.get("min_punctuated_utterance_ratio", 0)
    )
    missing_punctuation = [
        token
        for token in prosody_config.get("required_tokens", [])
        if punctuation_counts.get(token, 0) == 0
    ]
    if punctuated_ratio < minimum_punctuated or missing_punctuation:
        raise RuntimeError(
            "Source manifest cannot train punctuation: punctuated_ratio={:.2%} "
            "minimum={:.2%}, missing_required={}. Supply a punctuated manifest; "
            "MFA TextGrids remain reusable when normalized words are unchanged.".format(
                punctuated_ratio, minimum_punctuated, missing_punctuation
            )
        )
    resolved_rows = [
        (
            basename,
            _resolve_wav_path(wav_rel, base, manifest.parent),
            raw_text,
        )
        for basename, wav_rel, raw_text in rows
    ]
    missing_preflight = [
        {"id": basename, "reason": "missing_wav", "path": str(wav_path)}
        for basename, wav_path, _ in resolved_rows
        if not wav_path.is_file()
    ]
    maximum_ratio = float(
        config["preprocessing"].get("max_rejected_ratio", 0.01)
    )
    if rows and len(missing_preflight) / len(rows) > maximum_ratio:
        raise RuntimeError(
            "Wav path preflight failed before modifying raw output: {}/{} "
            "missing ({:.2%}, allowed {:.2%}). Examples: {}. "
            "Check path.corpus_path and path.manifest_path."
            .format(
                len(missing_preflight),
                len(rows),
                len(missing_preflight) / len(rows),
                maximum_ratio,
                missing_preflight[:10],
            )
        )

    accepted = []
    prosody_entries = {}
    rejected = []
    for basename, wav_path, raw_text in tqdm(resolved_rows):

        if not wav_path.is_file():
            rejected.append({"id": basename, "reason": "missing_wav", "path": str(wav_path)})
            continue

        text = normalize_for_mfa(raw_text)
        if not text:
            rejected.append({"id": basename, "reason": "empty_normalized_text"})
            continue

        wav, _ = librosa.load(str(wav_path), sr=sampling_rate)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if not np.isfinite(peak) or peak == 0.0:
            rejected.append({"id": basename, "reason": "silent_or_invalid_audio"})
            continue

        wav = wav / peak * max_wav_value
        wavfile.write(
            str(speaker_dir / "{}.wav".format(basename)),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(speaker_dir / "{}.lab".format(basename), "w", encoding="utf-8") as lab:
            lab.write(text)

        accepted.append(basename)
        punctuated = normalize_with_punctuation(raw_text)
        prosody_entries[basename] = {
            "text": punctuated,
            "mfa_text": text,
            "has_punctuation": any(token for _, token in prosody_words(raw_text)),
        }

    report = {
        "manifest": str(manifest),
        "speaker": speaker,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_ids": accepted,
        "rejected": rejected,
        "pruned_stale_output": False,
    }
    report_path = out_dir / "prepare_align_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    prosody_path = Path(
        config["path"].get("prosody_manifest_path", out_dir / "prosody_manifest.json")
    )
    prosody_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prosody_path, "w", encoding="utf-8") as prosody_file:
        json.dump(
            {
                "format_version": 1,
                "normalizer_version": NORMALIZER_VERSION,
                "source_manifest": str(manifest),
                "entries": prosody_entries,
            },
            prosody_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    rejected_ratio = len(rejected) / len(rows) if rows else 1.0
    if not accepted or rejected_ratio > maximum_ratio:
        raise RuntimeError(
            "Preparation rejected {}/{} rows ({:.2%}, allowed {:.2%}); "
            "stale raw output was NOT pruned. Inspect {}."
            .format(
                len(rejected), len(rows), rejected_ratio, maximum_ratio, report_path
            )
        )

    if config["preprocessing"].get("prune_raw_output", True):
        accepted_set = set(accepted)
        for pattern in ("*.wav", "*.lab"):
            for path in speaker_dir.glob(pattern):
                if path.stem not in accepted_set:
                    path.unlink()
        report["pruned_stale_output"] = True
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, ensure_ascii=False, indent=2)

    print("[prepare_align] accepted:", len(accepted))
    print("[prepare_align] rejected:", len(rejected))
    print("[prepare_align] report:", report_path)
    print("[prepare_align] prosody:", prosody_path)

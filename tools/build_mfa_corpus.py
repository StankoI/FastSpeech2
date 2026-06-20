#!/usr/bin/env python3
"""Build deterministic pseudo-speaker MFA input from prepared wav/lab pairs."""

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source, destination):
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="raw_data/Bulgarian/Bulgarian")
    parser.add_argument("--output", default="mfa_corpus")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if args.reset:
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    wavs = {path.stem: path for path in source.glob("*.wav")}
    labs = {path.stem: path for path in source.glob("*.lab")}
    if not wavs:
        raise SystemExit(
            "No prepared wav files found in {}. `prepare_align.py` must finish "
            "with accepted > 0 before building the MFA corpus.".format(source)
        )
    if set(wavs) != set(labs):
        raise SystemExit(
            "wav/lab mismatch: missing labs={} missing wavs={}".format(
                sorted(set(wavs) - set(labs))[:20],
                sorted(set(labs) - set(wavs))[:20],
            )
        )

    speakers = set()
    for uid in sorted(wavs):
        speaker = "__".join(uid.split("__")[:2]) or "Bulgarian"
        speakers.add(speaker)
        destination = output / speaker
        destination.mkdir(parents=True, exist_ok=True)
        link_or_copy(wavs[uid], destination / wavs[uid].name)
        link_or_copy(labs[uid], destination / labs[uid].name)

    report = {
        "source": str(source),
        "output": str(output),
        "utterances": len(wavs),
        "pseudo_speakers": len(speakers),
    }
    with open(output / "corpus_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

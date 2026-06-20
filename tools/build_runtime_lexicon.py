#!/usr/bin/env python3
"""Export highest-probability MFA dictionary pronunciations for inference."""

import argparse
import os
from pathlib import Path


def discover_dictionary(model_name):
    candidates = []
    if os.environ.get("MFA_ROOT_DIR"):
        candidates.append(
            Path(os.environ["MFA_ROOT_DIR"]) / "pretrained_models" / "dictionary" / (model_name + ".dict")
        )
    candidates.extend(
        [
            Path.home() / "Documents" / "MFA" / "pretrained_models" / "dictionary" / (model_name + ".dict"),
            Path.home() / "MFA" / "pretrained_models" / "dictionary" / (model_name + ".dict"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate {}. Pass --dictionary explicitly after `mfa model download dictionary {}`."
        .format(model_name, model_name)
    )


def parse_dictionary(path):
    best = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            phones = parts[-1].strip()
            if not word or not phones:
                continue
            try:
                probability = float(parts[1]) if len(parts) >= 3 else 1.0
            except ValueError:
                probability = 1.0
            if word not in best or probability > best[word][0]:
                best[word] = (probability, phones)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bulgarian_mfa")
    parser.add_argument("--dictionary")
    parser.add_argument(
        "--generated-oovs",
        help="plain G2P dictionary generated for corpus OOV words",
    )
    parser.add_argument("--overrides", default="lexicon/bulgarian_mfa_overrides.dict")
    parser.add_argument("--output", default="lexicon/bulgarian_mfa_runtime.dict")
    args = parser.parse_args()

    dictionary = Path(args.dictionary) if args.dictionary else discover_dictionary(args.model)
    entries = parse_dictionary(dictionary)
    if args.generated_oovs:
        generated = Path(args.generated_oovs)
        if not generated.is_file():
            raise FileNotFoundError(generated)
        entries.update(parse_dictionary(generated))
    overrides = Path(args.overrides)
    if overrides.is_file():
        entries.update(parse_dictionary(overrides))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for word in sorted(entries):
            handle.write("{}\t{}\n".format(word, entries[word][1]))
    print("Wrote {} pronunciations to {}".format(len(entries), output))


if __name__ == "__main__":
    main()

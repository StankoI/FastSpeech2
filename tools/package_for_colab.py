#!/usr/bin/env python3
"""Validate and package a completed local MFA phone run for Colab."""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/Bulgarian/preprocess.yaml")
    parser.add_argument("--output", default="mfa_phone_export")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "tools/validate_mfa_pipeline.py",
            "--config",
            args.config,
            "--stage",
            "alignment",
        ],
        check=True,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(
        str(output / "raw_data_Bulgarian"),
        "zip",
        root_dir="raw_data",
        base_dir="Bulgarian",
    )
    shutil.make_archive(
        str(output / "TextGrid_Bulgarian"),
        "zip",
        root_dir="preprocessed_data/Bulgarian",
        base_dir="TextGrid",
    )

    assets = output / "phoneme_assets.zip"
    required = [
        Path("lexicon/bulgarian_mfa_runtime.dict"),
        Path("text/bulgarian_mfa_phones.py"),
        Path("preprocessed_data/Bulgarian/TextGrid/alignment_run.json"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing package assets: {}".format(missing))
    with zipfile.ZipFile(assets, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in required:
            archive.write(path, arcname=str(path))

    print("Colab export ready in", output)
    for path in sorted(output.iterdir()):
        print("  {:32s} {:.1f} MB".format(path.name, path.stat().st_size / 1e6))


if __name__ == "__main__":
    main()


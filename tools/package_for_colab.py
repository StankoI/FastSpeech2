#!/usr/bin/env python3
"""Validate and package a completed local MFA phone run for Colab."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/Bulgarian/preprocess.yaml")
    parser.add_argument("--output", default="mfa_phone_export")
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="reuse existing raw/TextGrid archives and rebuild only small ABI assets",
    )
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="use a hash-bound marker from a just-completed alignment validation",
    )
    args = parser.parse_args()

    if args.reuse_validation:
        import yaml
        from text.bulgarian_mfa_phones import INVENTORY_VERSION

        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        grid_root = Path(config["path"]["preprocessed_path"]) / "TextGrid"
        marker_path = grid_root / "alignment_validation.json"
        if not marker_path.is_file():
            raise SystemExit("Missing validation marker; run validate_mfa_pipeline.py first")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        current = {
            "format_version": 1,
            "inventory_version": INVENTORY_VERSION,
            "wav_count": len(list(Path(config["path"]["raw_path"]).rglob("*.wav"))),
            "textgrid_count": len(list(grid_root.rglob("*.TextGrid"))),
            "prosody_manifest_sha256": hashlib.sha256(
                Path(config["path"]["prosody_manifest_path"]).read_bytes()
            ).hexdigest(),
            "alignment_run_sha256": hashlib.sha256(
                (grid_root / "alignment_run.json").read_bytes()
            ).hexdigest(),
            "dictionary_sha256": hashlib.sha256(
                Path(
                    config.get("mfa", {}).get("alignment_dictionary")
                    or config["path"]["runtime_lexicon_path"]
                ).read_bytes()
            ).hexdigest(),
        }
        if marker != current:
            raise SystemExit("Stale alignment validation marker: marker={} current={}".format(marker, current))
    else:
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
    if not args.assets_only:
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
    else:
        for name in ("raw_data_Bulgarian.zip", "TextGrid_Bulgarian.zip"):
            if not (output / name).is_file():
                raise SystemExit("--assets-only requires existing {}".format(output / name))

    assets = output / "phoneme_assets.zip"
    required = [
        Path("lexicon/bulgarian_mfa_runtime.dict"),
        Path("text/bulgarian_mfa_phones.py"),
        Path("raw_data/Bulgarian/prosody_manifest.json"),
        Path("bg_realdata/manifest_punctuated.csv"),
        Path("bg_realdata/punctuation_restore_report.json"),
        Path("preprocessed_data/Bulgarian/TextGrid/alignment_run.json"),
        Path("preprocessed_data/Bulgarian/TextGrid/alignment_validation.json"),
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

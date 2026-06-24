#!/usr/bin/env python3
"""Convenience wrapper for Bulgarian phoneme FastSpeech2 inference.

This script intentionally calls ``synthesize.py`` instead of reimplementing the
phonemization path. Raw Bulgarian text therefore goes through the same runtime
lexicon -> G2P cache -> MFA G2P fallback used by the training/inference code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from bg_text_normalizer import normalize as contextual_normalize
from bulgarian_normalization import normalize_with_punctuation


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"(\d+)\.pth\.tar", path.name)
    return int(match.group(1)) if match else None


def _latest_checkpoint(ckpt_dir: Path) -> tuple[int, Path]:
    checkpoints = []
    for path in ckpt_dir.glob("*.pth.tar"):
        step = _checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints like N.pth.tar found in {ckpt_dir}")
    return max(checkpoints, key=lambda item: item[0])


def _safe_output_id(text: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Keep filenames ASCII and boring; the real text is still passed to synthesize.py.
    ascii_words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    suffix = "_".join(ascii_words[:5])[:48]
    return f"infer_{timestamp}" + (f"_{suffix}" if suffix else "")


def _read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return args.text.strip()


def _normalize_text_for_inference(text: str, mode: str) -> str:
    """Normalize raw Bulgarian before the MFA-phone synthesis path.

    ``synthesize.py`` still performs the final punctuation/phone normalization.
    This step expands contextual constructs such as dates, currencies,
    measurements, ordinals, phones and common abbreviations before that final
    pass. Already-phonemized ``{...}`` input is intentionally left untouched.
    """
    if text.startswith("{") and text.endswith("}"):
        return text

    mode = mode.lower().strip()
    if mode == "none":
        return text
    if mode == "legacy":
        return normalize_with_punctuation(text)
    if mode == "contextual":
        return normalize_with_punctuation(contextual_normalize(text))
    raise ValueError("--text-normalizer must be 'contextual', 'legacy', or 'none'")


def _validate_generator_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        keys = list(checkpoint.keys())[:30] if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(
            f"{path} is not a HiFi-GAN generator checkpoint. Expected key "
            f"'generator', got keys/type: {keys}. For inference use g_..., not do_..."
        )


def _extract_default_vocoder(root: Path) -> Path:
    target = root / "hifigan" / "generator_universal.default.pth.tar"
    archive = root / "hifigan" / "generator_universal.pth.tar.zip"
    if target.is_file():
        return target
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing default HiFi-GAN archive: {archive}. Either keep the repo "
            "archive, or pass --vocoder-mode finetuned with --finetuned-vocoder."
        )
    with zipfile.ZipFile(archive) as zf:
        member = next(
            (
                name
                for name in zf.namelist()
                if name.endswith("generator_universal.pth.tar")
            ),
            None,
        )
        if member is None:
            raise FileNotFoundError(
                f"{archive} does not contain generator_universal.pth.tar"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination)
    _validate_generator_checkpoint(target)
    return target


def _prepare_vocoder(root: Path, mode: str, finetuned_vocoder: str | None) -> Path:
    runtime_path = root / "hifigan" / "generator_universal.pth.tar"
    mode = mode.lower().strip()

    if mode == "default":
        default_path = _extract_default_vocoder(root)
        shutil.copy2(default_path, runtime_path)
        active_path = default_path
    elif mode in {"finetuned", "custom"}:
        if not finetuned_vocoder:
            raise ValueError("--finetuned-vocoder is required for --vocoder-mode finetuned")
        active_path = Path(finetuned_vocoder).expanduser()
        if not active_path.is_absolute():
            active_path = (root / active_path).resolve()
        if not active_path.is_file():
            raise FileNotFoundError(active_path)
        _validate_generator_checkpoint(active_path)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active_path, runtime_path)
    else:
        raise ValueError("--vocoder-mode must be 'default' or 'finetuned'")

    _validate_generator_checkpoint(runtime_path)
    print(f"vocoder mode: {mode}")
    print(f"vocoder source: {active_path}")
    print(f"vocoder runtime: {runtime_path}")
    return active_path


def _resolve_checkpoint(
    train_config: dict,
    checkpoint: str | None,
    ckpt_dir: str | None,
    restore_step: int | None,
) -> tuple[int, Path]:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed_step = _checkpoint_step(path)
        if parsed_step is None:
            raise ValueError(
                "--checkpoint must be named like 22000.pth.tar because "
                "synthesize.py restores by numeric step."
            )
        if restore_step is not None and restore_step != parsed_step:
            raise ValueError(
                f"--restore-step={restore_step} does not match checkpoint {path.name}"
            )
        return parsed_step, path

    directory = Path(
        ckpt_dir or train_config["path"]["ckpt_path"]
    ).expanduser()
    if not directory.is_absolute():
        directory = (REPO_ROOT / directory).resolve()
    if restore_step is not None:
        path = directory / f"{restore_step}.pth.tar"
        if not path.is_file():
            raise FileNotFoundError(path)
        return restore_step, path
    return _latest_checkpoint(directory)


def _prepend_path(env: dict[str, str], directory: Path) -> None:
    directory = directory.expanduser().resolve()
    env["PATH"] = f"{directory}{os.pathsep}{env.get('PATH', '')}"


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.mfa_root_dir:
        env["MFA_ROOT_DIR"] = str(Path(args.mfa_root_dir).expanduser().resolve())
    if args.mamba_root_prefix:
        env["MAMBA_ROOT_PREFIX"] = str(
            Path(args.mamba_root_prefix).expanduser().resolve()
        )
    if args.mfa_bin:
        _prepend_path(env, Path(args.mfa_bin))
    else:
        mfa_path = Path(args.mfa_cmd)
        if mfa_path.is_absolute():
            _prepend_path(env, mfa_path.parent)
    return env


def _write_temp_train_config(
    train_config: dict,
    ckpt_dir: Path,
    result_dir: Path,
) -> Path:
    patched = dict(train_config)
    patched["path"] = dict(train_config.get("path", {}))
    patched["path"]["ckpt_path"] = str(ckpt_dir)
    patched["path"]["result_path"] = str(result_dir)
    patched["path"].setdefault("log_path", str(result_dir.parent / "log"))

    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".yaml", delete=False
    )
    with handle:
        yaml.safe_dump(patched, handle, allow_unicode=True, sort_keys=False)
    return Path(handle.name)


def _check_inference_assets(preprocess_config: dict, runtime_lexicon: str | None) -> None:
    preprocessed = Path(preprocess_config["path"]["preprocessed_path"])
    if not preprocessed.is_absolute():
        preprocessed = REPO_ROOT / preprocessed
    stats = preprocessed / "stats.json"
    if not stats.is_file():
        raise FileNotFoundError(
            f"Missing {stats}. For inference you need the A100 preprocessed "
            "assets restored, or at minimum stats.json in preprocessed_data/Bulgarian."
        )

    lexicon = Path(runtime_lexicon or preprocess_config["path"]["runtime_lexicon_path"])
    if not lexicon.is_absolute():
        lexicon = REPO_ROOT / lexicon
    if not lexicon.is_file():
        raise FileNotFoundError(
            f"Missing runtime lexicon {lexicon}. Restore phoneme_assets.zip first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bulgarian phoneme FastSpeech2 inference on raw text."
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Raw Bulgarian text or a {phone phone} sequence")
    text_group.add_argument("--text-file", help="UTF-8 file containing the text")

    parser.add_argument("--checkpoint", help="Exact checkpoint path, e.g. 22000.pth.tar")
    parser.add_argument("--ckpt-dir", help="Directory containing checkpoints")
    parser.add_argument(
        "--restore-step",
        type=int,
        help="Checkpoint step. If omitted, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--result-dir",
        default="output/result/Bulgarian",
        help="Directory for generated wav/png files",
    )
    parser.add_argument("--output-id", help="Output basename without extension")

    parser.add_argument(
        "-p",
        "--preprocess-config",
        default="config/Bulgarian/preprocess.yaml",
    )
    parser.add_argument("-m", "--model-config", default="config/Bulgarian/model.yaml")
    parser.add_argument("-t", "--train-config", default="config/Bulgarian/train.yaml")

    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument(
        "--duration-control",
        type=float,
        default=1.0,
        help="Larger values make speech slower; smaller values make it faster.",
    )
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--g2p-model", default="bulgarian_mfa")
    parser.add_argument(
        "--mfa-cmd",
        default="mfa",
        help="MFA executable name/path for OOV words",
    )
    parser.add_argument(
        "--mfa-bin",
        help="Directory containing MFA/OpenFST binaries; prepended to PATH",
    )
    parser.add_argument(
        "--mfa-root-dir",
        help="Optional MFA_ROOT_DIR cache/work directory",
    )
    parser.add_argument(
        "--mamba-root-prefix",
        help="Optional MAMBA_ROOT_PREFIX for micromamba-based MFA installs",
    )
    parser.add_argument("--runtime-lexicon", help="Override runtime lexicon path")
    parser.add_argument(
        "--text-normalizer",
        default="contextual",
        choices=["contextual", "legacy", "none"],
        help=(
            "Raw-text normalizer before phonemization. 'contextual' expands "
            "dates/currency/measurements/ordinals/phones/abbreviations; "
            "'legacy' uses the older number-only normalizer; 'none' passes "
            "raw text to synthesize.py."
        ),
    )
    parser.add_argument(
        "--vocoder-mode",
        default="default",
        choices=["default", "finetuned", "custom"],
        help="Use the repo universal HiFi-GAN or a fine-tuned generator checkpoint.",
    )
    parser.add_argument(
        "--finetuned-vocoder",
        help="Path to a HiFi-GAN generator checkpoint such as hifigan_finetune/g_00135000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    original_text = _read_text(args)
    text = _normalize_text_for_inference(original_text, args.text_normalizer)
    output_id = _safe_output_id(original_text, args.output_id)
    result_dir = Path(args.result_dir).expanduser()
    if not result_dir.is_absolute():
        result_dir = (REPO_ROOT / result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    preprocess_config_path = (REPO_ROOT / args.preprocess_config).resolve()
    model_config_path = (REPO_ROOT / args.model_config).resolve()
    train_config_path = (REPO_ROOT / args.train_config).resolve()

    with open(preprocess_config_path, encoding="utf-8") as handle:
        preprocess_config = yaml.safe_load(handle)
    with open(train_config_path, encoding="utf-8") as handle:
        train_config = yaml.safe_load(handle)

    _prepare_vocoder(REPO_ROOT, args.vocoder_mode, args.finetuned_vocoder)
    _check_inference_assets(preprocess_config, args.runtime_lexicon)

    step, checkpoint_path = _resolve_checkpoint(
        train_config,
        args.checkpoint,
        args.ckpt_dir,
        args.restore_step,
    )
    temporary_train_config = _write_temp_train_config(
        train_config,
        checkpoint_path.parent.resolve(),
        result_dir,
    )

    cmd = [
        sys.executable,
        "synthesize.py",
        "--restore_step",
        str(step),
        "--mode",
        "single",
        "--text",
        text,
        "--output_id",
        output_id,
        "--speaker_id",
        str(args.speaker_id),
        "--pitch_control",
        str(args.pitch_control),
        "--energy_control",
        str(args.energy_control),
        "--duration_control",
        str(args.duration_control),
        "--g2p_model",
        args.g2p_model,
        "--mfa_cmd",
        args.mfa_cmd,
        "-p",
        str(preprocess_config_path),
        "-m",
        str(model_config_path),
        "-t",
        str(temporary_train_config),
    ]
    if args.runtime_lexicon:
        cmd.extend(["--runtime_lexicon", args.runtime_lexicon])

    print(f"checkpoint: {checkpoint_path}")
    print(f"text normalizer: {args.text_normalizer}")
    if text != original_text:
        print(f"normalized text: {text}")
    print(f"output id:  {output_id}")
    print(f"result dir: {result_dir}")

    try:
        subprocess.run(cmd, check=True, env=_build_env(args))
    finally:
        try:
            temporary_train_config.unlink()
        except FileNotFoundError:
            pass

    wav = result_dir / f"{output_id}.wav"
    png = result_dir / f"{output_id}.png"
    print("generated:")
    if wav.is_file():
        print(f"  wav: {wav}")
    if png.is_file():
        print(f"  png: {png}")
    if not wav.is_file():
        raise FileNotFoundError(f"Expected output wav was not created: {wav}")


if __name__ == "__main__":
    main()

"""Reproducibility metadata for phoneme FastSpeech2 checkpoints."""

import hashlib
import json

from text.bulgarian_mfa_phones import INVENTORY_VERSION
from text.symbols import symbols, symbols_sha256


def _hash_json(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_metadata(preprocess_config, model_config):
    semantic_preprocess = {
        "mfa": preprocess_config.get("mfa", {}),
        "preprocessing": preprocess_config["preprocessing"],
    }
    return {
        "format_version": 1,
        "inventory_version": INVENTORY_VERSION,
        "symbols": list(symbols),
        "symbols_sha256": symbols_sha256(),
        "preprocess_sha256": _hash_json(semantic_preprocess),
        "model_config_sha256": _hash_json(model_config),
    }


def validate_checkpoint_metadata(saved, preprocess_config, model_config):
    if not saved:
        raise RuntimeError(
            "Checkpoint has no phoneme pipeline metadata. It is a legacy/grapheme "
            "checkpoint and cannot be loaded with the frozen MFA phone inventory."
        )
    current = checkpoint_metadata(preprocess_config, model_config)
    fields = (
        "inventory_version",
        "symbols_sha256",
        "preprocess_sha256",
        "model_config_sha256",
    )
    mismatches = {
        field: {"checkpoint": saved.get(field), "current": current.get(field)}
        for field in fields
        if saved.get(field) != current.get(field)
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint/pipeline incompatibility: {}".format(
                json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        )


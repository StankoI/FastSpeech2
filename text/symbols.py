"""Stable symbol ABI for Bulgarian phoneme FastSpeech2 checkpoints."""

import hashlib
import json

from text.bulgarian_mfa_phones import INVENTORY_VERSION, PHONES


_pad = "_"

# The order is checkpoint ABI.  Do not sort dynamically or derive it from one
# particular alignment subset.
symbols = [_pad] + ["@" + phone for phone in PHONES]


def symbols_sha256():
    payload = json.dumps(
        {"inventory_version": INVENTORY_VERSION, "symbols": symbols},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

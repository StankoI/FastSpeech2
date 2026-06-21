"""Map MFA word/phone tiers to the FastSpeech2 linguistic token ABI."""

import unicodedata

import numpy as np


def align_with_prosody(phone_tier, word_tier, expected_word_prosody, sampling_rate, hop_length):
    expected_words = [word for word, _ in expected_word_prosody]
    word_intervals = [
        (
            interval.start_time,
            interval.end_time,
            unicodedata.normalize("NFC", interval.text.strip().lower()),
        )
        for interval in word_tier._objects
        if interval.text.strip()
    ]
    aligned_words = [label for _, _, label in word_intervals]
    if aligned_words != expected_words:
        raise RuntimeError(
            "TextGrid/source word mismatch: aligned={!r} expected={!r}".format(
                aligned_words, expected_words
            )
        )

    items = []
    for interval in phone_tier._objects:
        start, end = interval.start_time, interval.end_time
        phone = unicodedata.normalize("NFC", interval.text.strip())
        if phone in {"", "sil"}:
            phone = "sp"
        duration = int(
            np.round(end * sampling_rate / hop_length)
            - np.round(start * sampling_rate / hop_length)
        )
        word_index = None
        if phone != "sp":
            overlaps = [
                max(0.0, min(end, word_end) - max(start, word_start))
                for word_start, word_end, _ in word_intervals
            ]
            if not overlaps or max(overlaps) <= 0:
                raise RuntimeError(
                    "Phone {!r} at {:.3f}-{:.3f} is outside every word interval".format(
                        phone, start, end
                    )
                )
            word_index = int(np.argmax(overlaps))
        items.append(
            {
                "phone": phone,
                "duration": duration,
                "start": start,
                "end": end,
                "word_index": word_index,
            }
        )

    lexical_positions = [i for i, item in enumerate(items) if item["phone"] != "sp"]
    if not lexical_positions:
        return [], [], 0, 0
    items = items[lexical_positions[0] : lexical_positions[-1] + 1]
    start_time, end_time = items[0]["start"], items[-1]["end"]

    merged = []
    for item in items:
        if item["phone"] == "sp" and merged and merged[-1]["phone"] == "sp":
            merged[-1]["duration"] += item["duration"]
            merged[-1]["end"] = item["end"]
        else:
            merged.append(dict(item))
    items = merged

    last_phone_for_word = {}
    seen_words = set()
    for index, item in enumerate(items):
        if item["word_index"] is not None:
            last_phone_for_word[item["word_index"]] = index
            seen_words.add(item["word_index"])
    expected_indices = set(range(len(expected_words)))
    if seen_words != expected_indices:
        raise RuntimeError(
            "Not every aligned word has a phone: missing={}".format(
                sorted(expected_indices - seen_words)
            )
        )

    controls_after = {}
    for word_index, (_, punctuation) in enumerate(expected_word_prosody):
        position = last_phone_for_word[word_index]
        if punctuation:
            if position + 1 < len(items) and items[position + 1]["phone"] == "sp":
                items[position + 1]["phone"] = punctuation
            else:
                controls_after.setdefault(position, []).append((punctuation, 0))
        elif word_index < len(expected_words) - 1:
            controls_after.setdefault(position, []).append(("wb", 0))

    phones, durations = [], []
    for index, item in enumerate(items):
        phones.append(item["phone"])
        durations.append(item["duration"])
        for token, duration in controls_after.get(index, []):
            phones.append(token)
            durations.append(duration)
    if len(phones) != len(durations) or any(not phone for phone in phones):
        raise RuntimeError("Invalid phone/duration alignment")
    return phones, durations, start_time, end_time

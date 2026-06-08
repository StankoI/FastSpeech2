import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def prepare_align(config):
    """Build the raw_data tree MFA + the preprocessor consume.

    Reads the cleaned manifest produced by clean_dataset.py (``id|wav_path|text``),
    resamples each clip to the target sampling rate, peak-normalises it, and
    writes ``{id}.wav`` + ``{id}.lab`` under ``raw_path/Bulgarian/``. The text is
    already normalised (lowercase Bulgarian words) so it is written verbatim.
    """
    base = config["path"]["corpus_path"]
    manifest = config["path"]["manifest_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    speaker = "Bulgarian"

    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)

    with open(manifest, encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            base_name, wav_rel, text = parts[0], parts[1], parts[2]

            wav_path = wav_rel if os.path.isabs(wav_rel) else os.path.join(base, wav_rel)
            if not os.path.exists(wav_path):
                continue

            wav, _ = librosa.load(wav_path, sr=sampling_rate)
            peak = np.max(np.abs(wav))
            if peak == 0:
                continue
            wav = wav / peak * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                "w",
                encoding="utf-8",
            ) as f1:
                f1.write(text)

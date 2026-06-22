"""Emit waveforms trimmed to match the GTA mels, for HiFi-GAN finetuning.

The preprocessor builds each mel from raw_wav[int(sr*start) : int(sr*end)] (leading/
trailing silence cut using the TextGrid) and then clips it to sum(duration) frames.
The wavs in raw_data are the *untrimmed* originals, so they are longer than - and
shifted from - the GTA mels. HiFi-GAN assumes wav and mel are frame-aligned, so feeding
the raw wavs trains on misaligned pairs (and crashes on short clips).

This reproduces the trim: for each utterance it reads `start` (onset of the first
non-silence phone) from the TextGrid and writes
    raw_wav[int(sr*start) : int(sr*start) + T*hop]
where T = number of frames in the GTA mel. The result is a 22.05 kHz int16 wav that is
frame-aligned 1:1 with gta_mels/<id>.npy.

Example:
    python tools/dump_gta_wavs.py -p config/Bulgarian/preprocess.yaml \
        --gta_dir gta_mels/Bulgarian \
        --raw_wav_dir /content/data/raw/Bulgarian/Bulgarian \
        --out_dir gta_wavs/Bulgarian
"""

import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from scipy.io import wavfile

from textgrid_utils import read_textgrid

# Matches the preprocessor: "", "sil" -> "sp", and "sp" itself is silence.
SIL = {"", "sil", "sp"}


def trim_start_seconds(tg_path):
    """Onset (seconds) of the first non-silence phone, or None if all silence."""
    tg = read_textgrid(tg_path)
    for iv in tg["phones"]._objects:
        if unicodedata.normalize("NFC", iv.text.strip()) not in SIL:
            return iv.start_time
    return None


def main(args):
    cfg = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    sr = cfg["preprocessing"]["audio"]["sampling_rate"]
    hop = cfg["preprocessing"]["stft"]["hop_length"]
    tg_dir = os.path.join(cfg["path"]["preprocessed_path"], "TextGrid", "Bulgarian")

    ids = []
    for fl in args.filelists.split(","):
        with open(os.path.join(cfg["path"]["preprocessed_path"], fl), encoding="utf-8") as f:
            ids += [ln.split("|", 1)[0] for ln in f if ln.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_skip = 0
    for utt in ids:
        mel_path = os.path.join(args.gta_dir, utt + ".npy")
        wav_path = os.path.join(args.raw_wav_dir, utt + ".wav")
        tg_path = os.path.join(tg_dir, utt + ".TextGrid")
        if not (os.path.isfile(mel_path) and os.path.isfile(wav_path) and os.path.isfile(tg_path)):
            n_skip += 1
            continue
        start = trim_start_seconds(tg_path)
        if start is None:
            n_skip += 1
            continue

        T = np.load(mel_path).shape[1]
        rate, wav = wavfile.read(wav_path)  # int16 @ sr
        assert rate == sr, "{}: sample rate {} != {}".format(utt, rate, sr)
        s0 = int(sr * start)
        seg = wav[s0 : s0 + T * hop]
        if len(seg) < T * hop:  # raw ran short -> pad tail to keep length == T*hop
            seg = np.pad(seg, (0, T * hop - len(seg)))
        wavfile.write(os.path.join(args.out_dir, utt + ".wav"), sr, seg.astype(np.int16))
        n_ok += 1
        if n_ok % 2000 == 0:
            print("  ...{} written".format(n_ok), end="\r")

    print("\nDone. {} wavs -> {} ({} skipped)".format(n_ok, args.out_dir, n_skip))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--preprocess_config", required=True)
    ap.add_argument("--gta_dir", required=True)
    ap.add_argument("--raw_wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--filelists", default="train.txt,val.txt")
    main(ap.parse_args())

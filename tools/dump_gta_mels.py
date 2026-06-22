"""Dump ground-truth-aligned (GTA) mel-spectrograms from a trained FastSpeech2.

These are FastSpeech2's *own* predicted post-net mels, generated in teacher-forcing
mode (GT pitch/energy/duration passed in), so every output is frame-aligned to the
real waveform. Feed them to HiFi-GAN's `--fine_tuning` mode paired with the original
22.05 kHz wavs to close the acoustic-model / vocoder mismatch.

Each utterance is saved as `<out_dir>/<basename>.npy` with shape [n_mels, T] -- the
layout HiFi-GAN's meldataset expects.

Example (Colab, features + ckpt on Drive):
    python tools/dump_gta_mels.py \
        --restore_step 60000 \
        -p config/Bulgarian/preprocess.yaml \
        -m config/Bulgarian/model.yaml \
        -t config/Bulgarian/train.yaml \
        --filelist train.txt \
        --out_dir gta_mels/Bulgarian
    # repeat with --filelist val.txt
"""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import dataset as dataset_module
from dataset import Dataset
from utils.model import get_model
from utils.tools import to_device


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.skip_abi_check:
        # The GTA dump reads only mel/pitch/energy/duration + train.txt/val.txt; it
        # never touches prosody_manifest.json. Bypass the punctuation-integrity gate
        # so a missing/lost manifest doesn't block the dump. get_model() still runs
        # validate_checkpoint_metadata(), which is the real correctness guarantee.
        dataset_module._validate_linguistic_abi = lambda *a, **k: None
        print("[dump] ABI check skipped (--skip_abi_check)")

    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    model = get_model(args, configs, device, train=False)

    # sort=False / drop_last=False / shuffle=False -> every utterance dumped exactly once.
    dataset = Dataset(
        args.filelist, preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    n_saved = 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            ids = batch[0]
            mel_lens = batch[7]
            with torch.no_grad():
                # batch[2:] == speakers, texts, src_lens, max_src_len, mels, mel_lens,
                # max_mel_len, p_targets, e_targets, d_targets. Passing the GT duration
                # targets makes the length regulator reproduce the GT frame count.
                output = model(*(batch[2:]))
            postnet_mel = output[1]  # [B, T, n_mels]
            for i, utt_id in enumerate(ids):
                length = mel_lens[i].item()
                mel = postnet_mel[i, :length].transpose(0, 1)  # [n_mels, T]
                np.save(
                    os.path.join(args.out_dir, "{}.npy".format(utt_id)),
                    mel.cpu().numpy().astype(np.float32),
                )
                n_saved += 1
        print("  ...{} dumped".format(n_saved), end="\r")

    print("\nDone. {} GTA mels written to {}".format(n_saved, args.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--train_config", type=str, required=True)
    parser.add_argument(
        "--filelist",
        type=str,
        default="train.txt",
        help="metadata file under preprocessed_path (train.txt / val.txt)",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--skip_abi_check",
        action="store_true",
        help="bypass the prosody-manifest integrity gate (safe for the dump; "
        "the checkpoint-metadata check in get_model still runs)",
    )
    main(parser.parse_args())

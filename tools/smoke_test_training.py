#!/usr/bin/env python3
"""Run one real FastSpeech2 forward/backward pass before a long training job."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss
from utils.tools import to_device


def load_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preprocess-config", default="config/Bulgarian/preprocess.yaml")
    parser.add_argument("-m", "--model-config", default="config/Bulgarian/model.yaml")
    parser.add_argument("-t", "--train-config", default="config/Bulgarian/train.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    preprocess_config = load_yaml(args.preprocess_config)
    model_config = load_yaml(args.model_config)
    train_config = load_yaml(args.train_config)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    # Dataset.collate_fn uses the configured training batch size internally.
    # A temporary copy lets this smoke test stay small even when A100 training
    # is configured for a much larger batch.
    smoke_train_config = dict(train_config)
    smoke_train_config["optimizer"] = dict(train_config["optimizer"])
    smoke_train_config["optimizer"]["batch_size"] = args.batch_size
    dataset = Dataset(
        "train.txt",
        preprocess_config,
        smoke_train_config,
        sort=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    batch_groups = next(iter(loader))
    if len(batch_groups) != 1:
        raise RuntimeError("Smoke loader produced an unexpected batch grouping")
    batch = to_device(batch_groups[0], device)

    model = FastSpeech2(preprocess_config, model_config).to(device).train()
    criterion = FastSpeech2Loss(preprocess_config, model_config).to(device)
    amp_enabled = bool(
        train_config["optimizer"].get("amp", False) and device.type == "cuda"
    )
    amp_name = train_config["optimizer"].get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16

    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = model(*batch[2:])
        losses = criterion(batch, output)
    losses[0].backward()

    if not all(torch.isfinite(loss).item() for loss in losses):
        raise RuntimeError("Non-finite loss in training smoke test")
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("Non-finite gradient in training smoke test")

    print("[PASS] training forward/backward")
    print("device:", device)
    print("AMP:", amp_name if amp_enabled else "disabled")
    print("batch:", len(batch[0]))
    print("max phones:", int(batch[5]))
    print("max mel frames:", int(batch[8]))
    print("losses:", [round(float(loss.detach()), 6) for loss in losses])


if __name__ == "__main__":
    main()

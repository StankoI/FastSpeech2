"""Seed a HiFi-GAN finetuning run from the bundled universal *generator*.

jik876/hifi-gan resumes from a `g_<step>` + `do_<step>` pair and only warm-starts
the generator when BOTH files exist. We ship only the universal generator
(`hifigan/generator_universal.pth.tar[.zip]`, key "generator") and have no matching
discriminator. This writes:

    <out_dir>/g_00000000   -> {"generator": <universal weights>}
    <out_dir>/do_00000000  -> fresh MPD/MSD + fresh optim states, steps=0, epoch=-1

so the generator warm-starts and the two discriminators train from scratch (the
normal way to finetune a generator you don't have a paired discriminator for).

Run it from inside the jik876/hifi-gan clone (it imports that repo's models):

    python seed_hifigan_finetune.py \
        --hifigan_repo . \
        --universal /content/FastSpeech2/hifigan/generator_universal.pth.tar.zip \
        --config /content/FastSpeech2/hifigan/config.json \
        --out_dir /content/cp_bg_finetune
"""

import argparse
import io
import json
import os
import sys
import zipfile

import torch


def load_universal_generator(path):
    """Return the universal generator state dict, transparently unzipping .zip."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".pth.tar"))
            buf = io.BytesIO(zf.read(inner))
        ckpt = torch.load(buf, map_location="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt["generator"]


def main(args):
    sys.path.insert(0, args.hifigan_repo)
    # jik876/hifi-gan modules
    from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
    from env import AttrDict

    with open(args.config) as f:
        h = AttrDict(json.load(f))

    os.makedirs(args.out_dir, exist_ok=True)

    # --- generator: warm-start from universal ---
    generator = Generator(h)
    generator.load_state_dict(load_universal_generator(args.universal))
    torch.save(
        {"generator": generator.state_dict()},
        os.path.join(args.out_dir, "g_00000000"),
    )

    # --- discriminators + optimizers: fresh ---
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )
    torch.save(
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": 0,
            "epoch": -1,
        },
        os.path.join(args.out_dir, "do_00000000"),
    )

    print("Seeded {}/g_00000000 and do_00000000".format(args.out_dir))
    print("Now run train.py with --checkpoint_path {} --fine_tuning True".format(args.out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hifigan_repo", type=str, default=".")
    parser.add_argument("--universal", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    main(parser.parse_args())

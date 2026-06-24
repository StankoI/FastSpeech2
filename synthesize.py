import argparse
import json

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from bulgarian_normalization import prosody_words

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_mfa_dictionary_line(line):
    """
    MFA dictionary lines are usually:
        WORD<TAB>PHONE PHONE PHONE
    or with probabilities:
        WORD<TAB>PROB<TAB>PHONE PHONE PHONE
    """
    line = line.strip()
    if not line:
        return None, None

    parts = line.split("\t")

    if len(parts) >= 3:
        word = parts[0].strip()
        phones = parts[-1].strip().split()
        return word, phones

    if len(parts) == 2:
        word = parts[0].strip()
        phones = parts[1].strip().split()
        return word, phones

    # Fallback for whitespace-separated output.
    pieces = line.split()
    if len(pieces) >= 2:
        return pieces[0], pieces[1:]

    return None, None


def _mfa_g2p_word_prons(words, g2p_model="bulgarian_mfa", mfa_cmd="mfa"):
    """
    Run MFA G2P over a unique word list and return:
        {word: [phone1, phone2, ...]}

    Requires:
        mfa model download g2p bulgarian_mfa
    """
    unique_words = sorted(set(w for w in words if w.strip()))
    if not unique_words:
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "words.txt")
        out_path = os.path.join(tmp, "g2p.dict")

        with open(in_path, "w", encoding="utf-8") as f:
            for w in unique_words:
                f.write(w + "\n")

        executable = shutil.which(mfa_cmd) if not os.path.isabs(mfa_cmd) else mfa_cmd
        if not executable or not os.path.exists(executable):
            raise RuntimeError(
                "MFA executable {!r} was not found. Pass --mfa_cmd with the "
                "executable inside your MFA conda environment.".format(mfa_cmd)
            )
        cmd = [executable, "g2p", in_path, g2p_model, out_path, "--clean", "--sorted"]
        subprocess.run(cmd, check=True)

        prons = {}
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                word, phones = _parse_mfa_dictionary_line(line)
                if word and phones and word not in prons:
                    prons[word] = phones

    missing = [w for w in unique_words if w not in prons]
    if missing:
        raise RuntimeError(
            "MFA G2P did not return pronunciations for: "
            + ", ".join(missing[:30])
        )

    return prons


def _load_runtime_lexicon(path):
    """Load one deterministic pronunciation per word from a plain MFA lexicon."""
    lexicon = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            word, phones = _parse_mfa_dictionary_line(line)
            if word and phones and word not in lexicon:
                lexicon[word] = phones
    if not lexicon:
        raise RuntimeError("Runtime lexicon is empty: {}".format(path))
    return lexicon


def _load_g2p_cache(path):
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {word: list(phones) for word, phones in data.items()}


def _save_g2p_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def preprocess_bulgarian(
    text,
    preprocess_config,
    g2p_model="bulgarian_mfa",
    mfa_cmd="mfa",
    runtime_lexicon=None,
):
    """
    Convert Bulgarian input to the same MFA phone-token format used during training.

    Accepted input:
    1. Already-phonemized:
        "{phone1 phone2 phone3}"

    2. Raw Bulgarian text:
        "Здравей, как си?"
       This is normalized and sent through MFA G2P.
    """
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        phones = text
        normalized_text = text
    else:
        word_prosody = prosody_words(text)
        if not word_prosody:
            raise ValueError("Input is empty after Bulgarian normalization")
        normalized_text = " ".join(word for word, _ in word_prosody)
        all_words = [word for word, _ in word_prosody]

        lexicon_path = runtime_lexicon or preprocess_config["path"].get(
            "runtime_lexicon_path"
        )
        if not lexicon_path or not os.path.isfile(lexicon_path):
            raise FileNotFoundError(
                "Runtime lexicon not found: {}. Run `python "
                "tools/build_runtime_lexicon.py` after downloading the MFA "
                "Bulgarian dictionary.".format(lexicon_path)
            )
        word_prons = _load_runtime_lexicon(lexicon_path)

        cache_path = Path(preprocess_config["path"]["preprocessed_path"]) / "g2p_cache.json"
        cache = _load_g2p_cache(cache_path)
        missing = sorted(set(all_words) - set(word_prons) - set(cache))
        if missing:
            generated = _mfa_g2p_word_prons(
                missing,
                g2p_model=g2p_model,
                mfa_cmd=mfa_cmd,
            )
            cache.update(generated)
            _save_g2p_cache(cache_path, cache)
        word_prons.update(cache)

        phone_list = []
        for index, (word, punctuation) in enumerate(word_prosody):
            phone_list.extend(word_prons[word])
            if punctuation:
                phone_list.append(punctuation)
            elif index < len(word_prosody) - 1:
                phone_list.append("wb")

        phones = "{" + " ".join(phone_list) + "}"

    print("Raw Text Sequence: {}".format(normalized_text))
    print("Phoneme Sequence: {}".format(phones))

    sequence = np.array(
        text_to_sequence(
            phones,
            preprocess_config["preprocessing"]["text"]["text_cleaners"],
        )
    )

    # Guard against silent unknown-phone dropping in text_to_sequence.
    expected_len = len(phones[1:-1].split()) if phones.startswith("{") else len(sequence)
    if len(sequence) != expected_len:
        raise RuntimeError(
            f"text_to_sequence length mismatch: got {len(sequence)}, "
            f"expected {expected_len}. This usually means some MFA phones "
            f"are missing from text/symbols.py."
        )

    return np.asarray(sequence, dtype=np.int64)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--output_id",
        type=str,
        default=None,
        help=(
            "output basename for single-sentence mode; defaults to the first "
            "100 characters of --text"
        ),
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--g2p_model",
        type=str,
        default="bulgarian_mfa",
        help="MFA G2P model for Bulgarian raw-text synthesis",
    )

    parser.add_argument(
        "--mfa_cmd",
        type=str,
        default="mfa",
        help="MFA command name/path",
    )
    parser.add_argument(
        "--runtime_lexicon",
        type=str,
        default=None,
        help="plain word-to-phone lexicon; defaults to preprocess config",
    )

    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        output_id = args.output_id or args.text[:100]
        ids = [output_id]
        raw_texts = [args.text]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "bg":
            texts = np.array([
                preprocess_bulgarian(
                    args.text,
                    preprocess_config,
                    g2p_model=args.g2p_model,
                    mfa_cmd=args.mfa_cmd,
                    runtime_lexicon=args.runtime_lexicon,
                )
            ])
        else:
            raise ValueError(
                "Unsupported language '{}'; only 'bg' is configured.".format(
                    preprocess_config["preprocessing"]["text"]["language"]
                )
            )
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

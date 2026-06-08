# FastSpeech 2 — Bulgarian single-speaker TTS

A trimmed fork of [ming024's FastSpeech 2](https://github.com/ming024/FastSpeech2)
([paper](https://arxiv.org/abs/2006.04558)) set up to train a **single-speaker
Bulgarian** voice from the *Euthymius* LibriVox corpus.

Bulgarian orthography is highly phonemic, so instead of a phoneme dictionary this
fork uses a **grapheme** approach: each Cyrillic letter is one token. Montreal
Forced Aligner (MFA) is still used for per-token durations, but its acoustic
model is **trained from scratch** on these graphemes (the pretrained IPA model is
not compatible).

## Pipeline

The whole flow runs in Google Colab — see
[`colab/Bulgarian_FastSpeech2_Colab.ipynb`](colab/Bulgarian_FastSpeech2_Colab.ipynb)
(setup → MFA → preprocess → train → synthesize).

| Step | Command / file |
|------|----------------|
| 1. Clean + merge the corpus | `python clean_dataset.py` → `filelists/euthymius_clean.csv` |
| 2. Build the grapheme dictionary | `python generate_bg_dict.py` → `lexicon/bulgarian-grapheme.txt` |
| 3. Resample + write `.wav`/`.lab` | `python prepare_align.py config/Bulgarian/preprocess.yaml` |
| 4. Align (train MFA from scratch) | `mfa train raw_data/Bulgarian lexicon/bulgarian-grapheme.txt bg_acoustic_model.zip --output_directory preprocessed_data/Bulgarian/TextGrid` |
| 5. Extract features | `python preprocess.py config/Bulgarian/preprocess.yaml` |
| 6. Train | `python train.py -p config/Bulgarian/preprocess.yaml -m config/Bulgarian/model.yaml -t config/Bulgarian/train.yaml` |
| 7. Synthesize | `python synthesize.py --text "Здравей" --restore_step N --mode single -p config/Bulgarian/preprocess.yaml -m config/Bulgarian/model.yaml -t config/Bulgarian/train.yaml` |

## Bulgarian-specific code

- `text/bulgarian.py` — grapheme G2P + text normalisation (single source of truth).
- `num2wordBg.py` — Bulgarian number-to-words expansion.
- `clean_dataset.py` / `generate_bg_dict.py` — dataset cleaning and dictionary build.
- `preprocessor/bulgarian.py`, `config/Bulgarian/` — preprocessing + configs.

The vocoder is the bundled HiFi-GAN **universal** model
(`hifigan/generator_universal.pth.tar.zip`, unzipped at runtime).

## TensorBoard

```
tensorboard --logdir output/log/Bulgarian
```

## Credits

FastSpeech 2 implementation by [ming024](https://github.com/ming024/FastSpeech2),
based on [xcmyz's FastSpeech](https://github.com/xcmyz/FastSpeech). See `LICENSE`.

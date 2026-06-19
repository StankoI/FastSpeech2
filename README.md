# FastSpeech 2 — Bulgarian single-speaker TTS

A trimmed fork of [ming024's FastSpeech 2](https://github.com/ming024/FastSpeech2)
([paper](https://arxiv.org/abs/2006.04558)) set up to train a **single-speaker
Bulgarian** voice from a LibriVox audiobook corpus (`RealData/merged_dataset/`).

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
| 1. Prepare the corpus | `RealData/prepare_realdata.ipynb` → `RealData/merged_dataset/manifest.csv` (single source of truth, `id\|wav\|text`) |
| 2. Build the grapheme dictionary | `python generate_bg_dict.py` → `lexicon/dictionary_full.txt` (from the manifest, so 0 OOV) |
| 3. Build the MFA corpus + resample | `python build_mfa_corpus.py` (alignment corpus) / `python prepare_align.py config/Bulgarian/preprocess.yaml` (22.05 kHz `.wav`/`.lab`) |
| 4. Align (acoustic model already trained) | `mfa align mfa_corpus_real lexicon/dictionary_full.txt bg_acoustic_model.zip preprocessed_data/Bulgarian/TextGrid/Bulgarian` |
| 5. Extract features | `python preprocess.py config/Bulgarian/preprocess.yaml` |
| 6. Train | `python train.py -p config/Bulgarian/preprocess.yaml -m config/Bulgarian/model.yaml -t config/Bulgarian/train.yaml` |
| 7. Synthesize | `python synthesize.py --text "Здравей" --restore_step N --mode single -p config/Bulgarian/preprocess.yaml -m config/Bulgarian/model.yaml -t config/Bulgarian/train.yaml` |

## Bulgarian-specific code

- `text/bulgarian.py` — grapheme G2P + text normalisation (single source of truth).
- `num2wordBg.py` — Bulgarian number-to-words expansion.
- `generate_bg_dict.py` / `build_mfa_corpus.py` — grapheme dictionary + MFA corpus, both built from the manifest. (`clean_dataset.py` is the legacy Euthymius-corpus cleaner.)
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

# FastSpeech 2 — Bulgarian single-speaker TTS

A trimmed fork of [ming024's FastSpeech 2](https://github.com/ming024/FastSpeech2)
([paper](https://arxiv.org/abs/2006.04558)) set up to train a **single-speaker
Bulgarian phoneme model** from a multi-audiobook corpus.

Text is normalized to synchronized word-only (MFA) and punctuation-preserving
(FastSpeech2) views. The model sees explicit word-boundary, comma, period,
question, exclamation, colon, semicolon, and dash tokens. Alignment uses the
official Bulgarian MFA dictionary and pretrained
acoustic model, with the Bulgarian G2P model as an OOV fallback. FastSpeech2's
phone embeddings themselves are trained from scratch.

## Pipeline

Run MFA locally, validate/package its output, then use
[`colab/Bulgarian_Phoneme_A100_Colab.ipynb`](colab/Bulgarian_Phoneme_A100_Colab.ipynb)
for feature extraction and A100 training. Full commands and recovery instructions
are in [`docs/PHONEME_PIPELINE.md`](docs/PHONEME_PIPELINE.md).

| Step | Command / file |
|------|----------------|
| 1. Restore real punctuation | `python tools/restore_punctuation_from_books.py --allow-fuzzy` |
| 2. Prepare wav/lab + prosody | `python prepare_align.py config/Bulgarian/preprocess.yaml` |
| 3. Build pseudo-speakers | `python tools/build_mfa_corpus.py --reset` |
| 4. Align with dictionary + G2P | `python tools/run_mfa_alignment.py --reset-output` |
| 5. Validate | `python tools/validate_mfa_pipeline.py --stage alignment` |
| 6. Package | `python tools/package_for_colab.py` |
| 7. Preprocess and train | `colab/Bulgarian_Phoneme_A100_Colab.ipynb` |
| 8. Synthesize | `python synthesize.py --text "Здравей, как си?" --restore_step N --mode single -p config/Bulgarian/preprocess.yaml -m config/Bulgarian/model.yaml -t config/Bulgarian/train.yaml` |

## Bulgarian-specific code

- `bulgarian_normalization.py` — shared word/punctuation normalization.
- `prosody_alignment.py` — injects word/punctuation controls into MFA durations.
- `text/bulgarian_mfa_phones.py` — frozen ordered phone inventory.
- `tools/run_mfa_alignment.py` — safe resumable G2P-backed alignment.
- `num2wordBg.py` — Bulgarian number-to-words expansion.
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

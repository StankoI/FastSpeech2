# Bulgarian single-speaker TTS with FastSpeech 2 and HiFi-GAN

This branch contains a Bulgarian text-to-speech pipeline based on FastSpeech 2
for acoustic modelling and HiFi-GAN for waveform synthesis. The project follows
the accompanying paper: a single-speaker Bulgarian corpus was prepared from
studio-recorded audiobooks, the text was normalized for Bulgarian-specific
non-standard words, phoneme durations were obtained with Montreal Forced
Aligner, and the final mel-spectrograms are converted to audio with HiFi-GAN.

The current branch is the punctuation-aware phoneme version. It uses a Bulgarian
runtime lexicon, MFA/G2P fallback for unknown words, explicit word-boundary and
punctuation tokens, and a contextual text normalizer for numbers, dates,
currencies, measurements, telephone numbers and common abbreviations.

## Quick Demo Through The Inference Notebook

Use this path when you only want to synthesize speech from an already trained
checkpoint.

### 1. Open the local inference notebook

From the repository root:

```bash
cd FastSpeech2
python -m pip install -r requirements.txt notebook
jupyter notebook notebooks/Bulgarian_Phoneme_Local_Inference.ipynb
```

If you are already in this repository checkout, open:

```text
notebooks/Bulgarian_Phoneme_Local_Inference.ipynb
```

Run the notebook cells from top to bottom. The first cell auto-detects the repo
root and imports the Bulgarian normalizer. The configuration cell controls the
checkpoint, vocoder and speaking-rate settings.

### 2. Required local files

For local inference the notebook expects these files or equivalent paths:

```text
output/ckpt/Bulgarian/*.pth.tar
lexicon/bulgarian_mfa_runtime.dict
preprocessed_data/Bulgarian/stats.json
hifigan/generator_universal.pth.tar.zip
```

Optional but recommended:

```text
hifigan_finetune/g_00135000
local_assets/phoneme_assets.zip
local_assets/preprocessed_Bulgarian_prosody_v2.zip
```

The notebook can unpack the two `local_assets/*.zip` archives automatically.
`phoneme_assets.zip` restores the runtime lexicon and phoneme assets.
`preprocessed_Bulgarian_prosody_v2.zip` restores the minimal preprocessed
metadata needed for inference.

If the trained checkpoint is not in `output/ckpt/Bulgarian`, set `CHECKPOINT`
to an exact file such as:

```python
CHECKPOINT = "/path/to/60000.pth.tar"
```

or set `CKPT_DIR` to the directory containing numbered checkpoints.

### 3. Choose vocoder and text

The final notebook cell contains the text and calls inference:

```python
text = "Аз съм говорещ модел и съм правен от студенти 3-ти и 4-ти курс"
wav_default = run_inference("finetuned", text)
```

Use:

```python
run_inference("default", text)
```

for the bundled universal HiFi-GAN vocoder, or:

```python
run_inference("finetuned", text)
```

for the Bulgarian fine-tuned generator. The fine-tuned vocoder should point to a
generator checkpoint named like `g_00135000`, not a discriminator checkpoint
named like `do_00135000`.

The main controls are:

```python
DURATION_CONTROL = 1.15  # larger means slower speech
PITCH_CONTROL = 1.0
ENERGY_CONTROL = 1.0
TEXT_NORMALIZER_MODE = "contextual"  # contextual | legacy | none
```

Generated audio and spectrogram plots are written to:

```text
local_inference_results/
```

### 4. Unknown words and MFA/G2P

Known words are resolved through `lexicon/bulgarian_mfa_runtime.dict` and the
packaged G2P cache. If a sentence contains a word outside both, install or point
the notebook to Montreal Forced Aligner with the Bulgarian G2P model:

```python
MFA_CMD = "mfa"
MFA_BIN = None
MFA_ROOT_DIR = None
MAMBA_ROOT_PREFIX = None
```

For a custom MFA installation, set `MFA_CMD` or `MFA_BIN` in the configuration
cell. The Colab inference notebook can install an MFA/G2P environment
automatically.

## Colab Inference

Use this when the checkpoint and assets are in Google Drive from the A100
training notebook.

Open:

```text
colab/Bulgarian_Phoneme_Inference_Colab.ipynb
```

In the setup cell set:

```python
REPO_URL = "https://github.com/<your-user>/FastSpeech2.git"
BRANCH = "phoneme-mfa"
DRIVE_DIR = "/content/drive/MyDrive/fs2_bg_phone"
```

The Drive directory should contain:

```text
phoneme_assets.zip
preprocessed_Bulgarian_prosody_v2.zip
output_prosody_v2/ckpt/Bulgarian/*.pth.tar
```

Optional for better sound quality:

```text
hifigan_finetune/g_00135000
```

Run the Colab notebook top to bottom once. Afterwards, to synthesize another
sentence, change only `TEXT` and rerun the final inference cell. Results are
written to:

```text
<DRIVE_DIR>/inference_results/
```

## Direct CLI Inference

The notebooks call the same wrapper, so the equivalent command is:

```bash
python tools/infer_bulgarian.py \
  --text "Здравейте! Това е демонстрация на български синтез на реч." \
  --ckpt-dir output/ckpt/Bulgarian \
  --result-dir local_inference_results \
  --duration-control 1.15 \
  --text-normalizer contextual \
  --vocoder-mode finetuned \
  --finetuned-vocoder hifigan_finetune/g_00135000
```

For the universal vocoder, replace the last two arguments with:

```bash
--vocoder-mode default
```

## Training Pipeline

The full training path is documented in
[`docs/PHONEME_PIPELINE.md`](docs/PHONEME_PIPELINE.md). In short:

| Step | Command / file |
| --- | --- |
| Restore punctuation | `python tools/restore_punctuation_from_books.py --allow-fuzzy` |
| Prepare wav/lab/prosody | `bash tools/prepare_mfa_inputs.sh` |
| Align with MFA | `python tools/run_mfa_alignment.py --reset-output` |
| Validate alignment | `python tools/validate_mfa_pipeline.py --stage alignment` |
| Package for Colab | `python tools/package_for_colab.py` |
| Train on A100 | `colab/Bulgarian_Phoneme_A100_Colab.ipynb` |
| Run inference | `notebooks/Bulgarian_Phoneme_Local_Inference.ipynb` or `colab/Bulgarian_Phoneme_Inference_Colab.ipynb` |

Punctuation-aware checkpoints are not compatible with the older grapheme or
punctuationless checkpoints because the phone/token ABI is different.

## Repository Layout

```text
bg_text_normalizer/            Bulgarian contextual text normalization
bulgarian_normalization.py     Shared punctuation and MFA text normalization
prosody_alignment.py           Word-boundary and punctuation duration injection
text/bulgarian_mfa_phones.py   Ordered Bulgarian phone inventory
tools/infer_bulgarian.py       CLI wrapper used by the inference notebooks
tools/run_mfa_alignment.py     Resumable MFA alignment runner
tools/package_for_colab.py     Archives assets for Colab training/inference
config/Bulgarian/              FastSpeech 2 Bulgarian configs
notebooks/                     Local demo/inference notebooks
colab/                         Colab training and inference notebooks
hifigan/                       HiFi-GAN code and bundled universal checkpoint zip
```

## Notes From The Paper

The submitted paper describes the complete project: approximately 70 hours of
single-speaker Bulgarian audiobook recordings were segmented into short
utterances, normalized into pronounceable Bulgarian text, aligned at phoneme
level with MFA, and used to train FastSpeech 2 from scratch. The strongest
results came from the cleaned phoneme-based run. Fine-tuning HiFi-GAN on the
target Bulgarian voice reduced the metallic artifacts of the universal vocoder
and is the recommended setting for final demos.

## Credits

The acoustic model is based on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). The project uses
FastSpeech 2, HiFi-GAN and Montreal Forced Aligner as described in the paper and
in the references there.

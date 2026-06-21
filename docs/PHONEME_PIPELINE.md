# Bulgarian MFA phoneme pipeline

This is the canonical workflow for the phoneme branch. The older grapheme MFA
notebooks are not compatible with these checkpoints.

## 1. Local prerequisites

Create an MFA environment and install the project preprocessing dependencies:

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner librosa tqdm pyyaml -y
conda run -n mfa mfa model download dictionary bulgarian_mfa
conda run -n mfa mfa model download acoustic bulgarian_mfa
conda run -n mfa mfa model download g2p bulgarian_mfa
conda run -n mfa mfa model inspect g2p bulgarian_mfa
```

If the last command cannot find the G2P model, list the versions available to
your exact MFA installation and install the Bulgarian MFA v2.0.0 G2P release.
Do not start alignment until `model inspect g2p` succeeds.

The default config expects:

```text
bg_realdata/manifest.csv
bg_realdata/wavs/<utterance-id>.wav
```

Every manifest row must be `id|wav_path|Bulgarian text`. **The third field must
contain the real source punctuation.** The current `bg_realdata/manifest.csv`
has already had every punctuation mark stripped and intentionally fails the
prosody gate; it is not valid punctuation-training input. The checked-in config
uses `corpus_path: bg_realdata`, so a manifest path such as `wavs/clip.wav`
resolves to `bg_realdata/wavs/clip.wav`. Resolution also falls back to the
manifest's own directory. A preflight aborts before pruning output if more than
1% of wav paths are missing.

The configured `bg_realdata/manifest_punctuated.csv` is reconstructed from the
five original books with:

```bash
python tools/restore_punctuation_from_books.py --allow-fuzzy
```

The tool uses both `dataset_full.csv` and `dropped.csv` as chronological
anchors, writes `punctuation_restore_report.json`, and requires every recovered
row to normalize back to the unchanged MFA words. The strict gate requires at least
5% punctuated utterances and occurrences of comma, period, question, and
exclamation tokens. It never guesses punctuation from acoustic silence.

For newly cut audiobooks, rerun `testSplit/split_by_sentences.py` from the
original punctuated transcript. It now sends a word-only view to aeneas but
writes the punctuation-preserving view to `metadata.csv`; the former behavior
that wrote stripped text into metadata was the original source of this loss.

If only punctuation changed, existing `.lab` files and TextGrids remain valid.
Build the sidecar and verify exact word identity without rewriting the WAVs:

```bash
python tools/build_prosody_manifest.py
```

If this reports a `.lab` mismatch, words changed too; use the full preparation
and alignment path below. `--allow-unpunctuated` exists only for diagnosing a
word-boundary-only sidecar and is rejected by the training config.

## 2. Prepare wav/lab and the runtime dictionary

Run from the repository root:

```bash
bash tools/prepare_mfa_inputs.sh
```

The script uses `set -e`: if preparation fails, G2P and lexicon generation do
not run on an empty corpus. Set `MFA_ENV=another_name` if your conda environment
is not named `mfa`.

Preparation writes `prepare_align_report.json` and `prosody_manifest.json`.
MFA `.lab` files contain words only; the sidecar retains normalized punctuation.
Inspect the report. Rejected inputs are
explicit and are not allowed to appear in the MFA corpus. Once preparation
finishes, stale wav/lab pairs not present in the accepted manifest are pruned.

## 3. Fresh G2P-backed alignment

The old TextGrids contain lexical `spn` and stale chunk markers. The first phone
run must therefore explicitly reset only the alignment output:

```bash
conda run --no-capture-output -n mfa python tools/run_mfa_alignment.py \
  --mfa-command mfa \
  --dictionary lexicon/bulgarian_mfa_runtime.dict \
  --acoustic bulgarian_mfa \
  --g2p bulgarian_mfa \
  --jobs 2 \
  --max-utterances 3000 \
  --reset-output
```

On later restarts run the same command without `--reset-output`. Chunk IDs are
derived from their utterance lists and the whole run is tied to the corpus,
models and MFA version. Changing a semantic input causes a clear refusal instead
of silently reusing old markers.

Alignment is accepted only when every wav has one TextGrid. The runtime/alignment
dictionary contains the official highest-probability pronunciations, generated
pronunciations for every corpus OOV, and any manual overrides. MFA also receives
`--g2p_model_path bulgarian_mfa` as a final safety net. The same plain dictionary
is packaged for inference, so known words have identical train/test phones.

MFA can occasionally refuse a corrupt or acoustically unusable utterance. The
runner allows at most 1% explicit export failures, records their exact IDs in
`alignment_run.json`, and downstream validation/preprocessing excludes only
those IDs. Raising `--max-failed-ratio` is not a substitute for investigating a
large failure count.

## 4. Mandatory local validation

```bash
conda run --no-capture-output -n mfa python tools/validate_mfa_pipeline.py \
  --config config/Bulgarian/preprocess.yaml \
  --stage alignment
```

The command must print PASS and nonzero source-punctuation counts. It verifies
that sidecar words, `.lab` words, and every TextGrid word tier are identical.
By default even one normal word aligned as `spn`
fails the run. Genuine annotated noise can be tolerated explicitly with
`--allow-spn-words N`, but this must not be used to hide dictionary OOVs.

## 5. Package for Colab

```bash
conda run --no-capture-output -n mfa python tools/package_for_colab.py \
  --output mfa_phone_export
```

When the existing raw and TextGrid ZIPs are already valid and only the prosody
sidecar/ABI changed, avoid duplicating 8+ GB and rebuild only the small archive:

```bash
python tools/package_for_colab.py --output mfa_phone_export \
  --assets-only --reuse-validation
```

`--reuse-validation` accepts only the hash-bound marker written by the preceding
alignment validation; changing the sidecar, dictionary, alignment metadata, ABI,
or corpus counts invalidates it.

Upload these files to `MyDrive/fs2_bg_phone/`:

- `raw_data_Bulgarian.zip`
- `TextGrid_Bulgarian.zip`
- `phoneme_assets.zip`

The assets archive includes the recovered punctuated manifest, its exact/fuzzy
matching audit report, the generated prosody sidecar, runtime lexicon, and ABI
inventory.

Then use `colab/Bulgarian_Phoneme_A100_Colab.ipynb`. MFA does not run on the A100;
the expensive GPU runtime is used only for feature extraction and training.

Before opening Colab, commit and push the complete phoneme branch to your fork.
Set `REPO_URL` in the notebook to that fork.  The notebook stores the first
training commit in `MyDrive/fs2_bg_phone/repo_commit_prosody_v2.txt` and automatically
checks out that exact commit when resuming.

The first Colab session runs in this order:

1. clone the pinned branch and install preprocessing dependencies;
2. copy the three archives from Drive to the VM and unpack them locally;
3. validate the alignment snapshot;
4. preprocess and validate all generated ABI-v2 features;
5. create and verify `preprocessed_Bulgarian_prosody_v2.zip` in Drive;
6. run the real forward/backward smoke test;
7. link `output/` to Drive and start training at step zero.

On later sessions, restore `preprocessed_Bulgarian_prosody_v2.zip` instead of the
three alignment archives.  Re-run the identical training configuration and
smoke-test cells, recreate the output link, and use the resume cell.  It checks
candidate checkpoints newest-first and skips an incomplete file.

## 6. Local preprocessing alternative

If preprocessing locally instead of Colab:

```bash
python preprocess.py config/Bulgarian/preprocess.yaml
python tools/validate_mfa_pipeline.py --stage preprocessed
```

`preprocess.py` intentionally clears old mel/pitch/energy/duration directories
before rebuilding them. It preserves the TextGrids. Empty MFA phone intervals
are mapped to the canonical `sp` token; leading/trailing pauses are trimmed.
`wb` is inserted between ordinary words with duration 0. A true punctuation
token consumes the following aligned silence when present, otherwise it has
duration 0. Punctuation injection therefore never changes the mel-frame sum.

## 7. Training and compatibility

Punctuation-aware training must start from step 0. Grapheme checkpoints and
older punctuationless phoneme checkpoints have different embedding ABIs and
are rejected. Old feature ZIPs are rejected because they lack
`linguistic_abi.json` and `wb`. Every new checkpoint stores hashes of:

- the ordered phone symbols;
- preprocessing semantics;
- model configuration.
- the exact `linguistic_abi.json`, which includes the prosody-manifest SHA-256.

Resume with exactly the same YAML values. A mismatch stops with an explanation
instead of loading incompatible weights.

At inference the same normalizer emits phones plus exact `wb`/punctuation
tokens, so punctuation is no longer reduced to generic `sp`. Raw-text inference
first uses the packaged runtime dictionary. A word absent
from that dictionary needs the Bulgarian MFA G2P model and an `mfa` executable;
pass its path with `--mfa_cmd`.  Already phonemized `{phone phone ...}` input and
sentences whose words are all in the runtime dictionary do not need MFA.

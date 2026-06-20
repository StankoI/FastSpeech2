#!/usr/bin/env bash
set -euo pipefail

MFA_ENV="${MFA_ENV:-mfa}"

conda run --no-capture-output -n "$MFA_ENV" \
  python prepare_align.py config/Bulgarian/preprocess.yaml

conda run --no-capture-output -n "$MFA_ENV" \
  python tools/build_mfa_corpus.py --reset

mkdir -p mfa_phone_work

conda run --no-capture-output -n "$MFA_ENV" \
  mfa g2p \
  mfa_corpus \
  bulgarian_mfa \
  mfa_phone_work/corpus_oovs.dict \
  --dictionary_path bulgarian_mfa \
  --clean \
  --sorted

conda run --no-capture-output -n "$MFA_ENV" \
  python tools/build_runtime_lexicon.py \
  --generated-oovs mfa_phone_work/corpus_oovs.dict

echo "[PASS] MFA corpus and runtime lexicon are ready"

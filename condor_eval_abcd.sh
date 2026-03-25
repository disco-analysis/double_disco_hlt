#!/bin/bash
set -e

echo "==== Job started: $(date) ===="
echo "Host: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "Torch CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda, "| CUDA available:", torch.cuda.is_available())')"

cd /afs/cern.ch/user/e/escheull/nobackup/modularized_nurd_hlt_con_ae

$PYTHON -m pip install -e . --quiet

export WANDB_API_KEY=$(cat ~/.wandb_api_key)

# find the latest checkpoint in ssl_checkpoints
LATEST_CKPT=/eos/user/e/escheull/ssl_checkpoints/embedding_hlt_linformer_supcon_encoder_20260318_092447.pth
echo "Using checkpoint: $LATEST_CKPT"

$PYTHON eval_abcd.py \
    --contrast_ckpt    "$LATEST_CKPT" \
    --contrast_test_pt /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --signal_pt        /eos/user/e/escheull/signal_pt/hlt_signal_TpTp.pt \
    --outdir           /eos/user/e/escheull/abcd_outputs

echo "==== Job finished: $(date) ===="

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install package in editable mode
$PYTHON -m pip install -e . --quiet

# Set up checkpoints symlink to EOS
EOS_CHECKPOINTS="/eos/user/e/escheull/ssl_checkpoints"
mkdir -p "$EOS_CHECKPOINTS"
if [ -d checkpoints ] && [ ! -L checkpoints ]; then
    cp -r checkpoints/* "$EOS_CHECKPOINTS/" 2>/dev/null || true
    rm -rf checkpoints
fi
if [ ! -L checkpoints ]; then
    ln -s "$EOS_CHECKPOINTS" checkpoints
fi

export WANDB_API_KEY=$(cat ~/.wandb_api_key)

$PYTHON train.py \
    --train_cfg configs/train_sup_con.yaml \
    --data_cfg  configs/data_smcocktail_withZB.yaml \
    --data      /eos/user/e/escheull/smcocktail_1M_withZB/hlt_smcocktail_train.pt

echo "==== Job finished: $(date) ===="

echo "Submitting eval_abcd job..."
condor_submit /afs/cern.ch/user/e/escheull/nobackup/modularized_nurd_hlt_con_ae/condor_eval_abcd.sub
echo "eval_abcd job submitted."

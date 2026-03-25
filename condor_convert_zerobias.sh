#!/bin/bash
set -e

echo "==== Job started: $(date) ===="
echo "Host: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"

cd /afs/cern.ch/user/e/escheull/nobackup/modularized_nurd_hlt_con_ae

$PYTHON -m pip install -e . tqdm --quiet

$PYTHON /afs/cern.ch/user/e/escheull/converterHLT.py \
    --config configs/data_zerobias.yaml \
    --overwrite

echo "==== Job finished: $(date) ===="

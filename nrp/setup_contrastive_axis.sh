#!/bin/bash
# Run this inside the utility pod:
#   kubectl exec -it contrastive-setup -n axol1tl -- bash /axovol/contrastive_axis/setup_contrastive_axis.sh
set -e

DEST=/axovol/contrastive_axis
mkdir -p "$DEST"
cd "$DEST"

echo "=== 1. Clone code ==="
if [ ! -d "modularized_nurd_hlt_con_ae" ]; then
    git clone --branch test_2 https://github.com/ellisonscheuller/modularized_nurd_hlt_con_ae.git
else
    echo "Repo already cloned, pulling latest..."
    git -C modularized_nurd_hlt_con_ae pull
fi

echo "=== 2. Install Python environment ==="
# Install to a PVC-backed directory so packages survive pod restarts
PYLIB="$DEST/pylib"
mkdir -p "$PYLIB"
export PYTHONPATH="$PYLIB:$PYTHONPATH"

pip install --quiet --target "$PYLIB" \
    torch==2.3.1 \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    wandb \
    PyYAML
pip install --quiet --target "$PYLIB" -e modularized_nurd_hlt_con_ae/

echo "# Add this to your shell or job script to use the persistent env:"
echo "export PYTHONPATH=$PYLIB:\$PYTHONPATH"

echo "=== 3. Data ==="
# Data is transferred separately from EOS via ssh pipe on the local machine.
# From your LOCAL machine run:
#
#   kubectl exec -n axol1tl contrastive-setup -- mkdir -p /axovol/contrastive_axis/data/signal_pt
#
#   ssh escheull@lxplus.cern.ch "cat /eos/user/e/escheull/smcocktail_1M_withZB/hlt_smcocktail_train.pt" \
#     | kubectl exec -i contrastive-setup -n axol1tl -- bash -c "cat > /axovol/contrastive_axis/data/hlt_smcocktail_train.pt"
#
#   ssh escheull@lxplus.cern.ch "cat /eos/user/e/escheull/smcocktail_1M_withZB/hlt_smcocktail_test.pt" \
#     | kubectl exec -i contrastive-setup -n axol1tl -- bash -c "cat > /axovol/contrastive_axis/data/hlt_smcocktail_test.pt"
#
#   ssh escheull@lxplus.cern.ch "cat /eos/user/e/escheull/signal_pt/hlt_signal_TpTp.pt" \
#     | kubectl exec -i contrastive-setup -n axol1tl -- bash -c "cat > /axovol/contrastive_axis/data/signal_pt/hlt_signal_TpTp.pt"
mkdir -p data/signal_pt
echo "Skipping data download — transfer via ssh pipe from local machine (see comments above)."
echo "Check existing files:"
ls -lh data/ 2>/dev/null || true

echo ""
echo "=== Setup complete! ==="
echo "Contents of $DEST:"
ls -lh "$DEST"
echo ""
echo "Data files:"
ls -lh "$DEST/data/"

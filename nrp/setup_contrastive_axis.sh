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

echo "=== 3. Download data from CERNBox (requires rclone config) ==="
# Install rclone if needed
if ! command -v rclone &> /dev/null; then
    curl -fsSL https://rclone.org/install.sh | bash
fi

mkdir -p data

# Check rclone config exists
if ! rclone listremotes | grep -q "cernbox:"; then
    echo ""
    echo "ERROR: rclone 'cernbox' remote not configured."
    echo "Run the following ONCE to set it up:"
    echo ""
    echo "  rclone config create cernbox webdav \\"
    echo "    url https://cernbox.cern.ch/remote.php/dav/files/escheull \\"
    echo "    vendor other \\"
    echo "    user escheull \\"
    echo "    pass \$(rclone obscure YOUR_CERNBOX_APP_PASSWORD)"
    echo ""
    echo "Get an app password at: https://cernbox.cern.ch/cernbox/desktop-app-setting"
    echo "Then re-run this script."
    exit 1
fi

echo "Downloading training data (withZB, ~16 GB)..."
rclone copy cernbox:smcocktail_1M_withZB/hlt_smcocktail_train.pt data/ --progress
rclone copy cernbox:smcocktail_1M_withZB/hlt_smcocktail_test.pt   data/ --progress

echo "Downloading signal data (~2.7 GB)..."
mkdir -p data/signal_pt
rclone copy cernbox:signal_pt/hlt_signal_TpTp.pt data/signal_pt/ --progress

echo ""
echo "=== Setup complete! ==="
echo "Contents of $DEST:"
ls -lh "$DEST"
echo ""
echo "Data files:"
ls -lh "$DEST/data/"

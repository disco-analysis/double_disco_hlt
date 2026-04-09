# NRP Setup — contrastive_axis

Target: `axol1tl` namespace, `axovol` PVC, folder `/axovol/contrastive_axis/`

## Step 1 — Copy this folder to the PVC

From your **local machine** (requires kubectl configured for axol1tl):

```bash
# Spin up utility pod
kubectl apply -f utility-pod.yaml -n axol1tl

# Wait for it to be Running
kubectl get pod contrastive-setup -n axol1tl -w

# Create destination directory and copy the setup script
kubectl exec -n axol1tl contrastive-setup -- mkdir -p /axovol/contrastive_axis
kubectl cp setup_contrastive_axis.sh axol1tl/contrastive-setup:/axovol/contrastive_axis/
```

## Step 2 — Configure CERNBox credentials (inside the pod)

```bash
kubectl exec -it contrastive-setup -n axol1tl -- bash
```

Inside the pod:
1. Go to https://cernbox.cern.ch/cernbox/desktop-app-setting and generate an **app password**
2. Run:
```bash
rclone config create cernbox webdav \
  url https://cernbox.cern.ch/remote.php/dav/files/escheull \
  vendor other \
  user escheull \
  pass $(rclone obscure YOUR_APP_PASSWORD)
```
3. Test it: `rclone ls cernbox:smcocktail_1M_withZB/`

## Step 3 — Run the setup script

```bash
kubectl exec -it contrastive-setup -n axol1tl -- bash /axovol/contrastive_axis/setup_contrastive_axis.sh
```

This will:
- Clone the repo (`test_2` branch) from GitHub
- Install the minimal Python environment (torch, scipy, sklearn, matplotlib, wandb, yaml)
- Download the .pt data files from CERNBox (~19 GB total):
  - `smcocktail_1M_withZB/hlt_smcocktail_train.pt`
  - `smcocktail_1M_withZB/hlt_smcocktail_test.pt`
  - `signal_pt/hlt_signal_TpTp.pt`

## Step 4 — Clean up pod

```bash
kubectl delete pod contrastive-setup -n axol1tl
```

## Final layout on PVC

```
/axovol/contrastive_axis/
  modularized_nurd_hlt_con_ae/   ← code (branch: test_2)
  data/
    hlt_smcocktail_train.pt
    hlt_smcocktail_test.pt
    signal_pt/
      hlt_signal_TpTp.pt
  setup_contrastive_axis.sh
```

## Notes

- The pip install inside the pod is ephemeral — if the pod restarts, re-install.
  Consider adding a `requirements.txt` inside the repo and installing to `/axovol/contrastive_axis/venv/`
  for persistence across pod restarts.
- WandB API key: `export WANDB_API_KEY=<your key>` inside the pod, or create a k8s secret.

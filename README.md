# Double DisCo HLT Anomaly Detection

An anomaly detection analysis using a two-axis ABCD background estimation method. Axis 1 is the AE reconstruction loss (how anomalous an event looks) and axis 2 is a contrastive model (Axo HLT model) score (what type of event it is). DisCo is used to decorrelate the two axes so the ABCD method is valid.

---

### Axis 1 — Autoencoder (density estimation)

An MLP autoencoder trained on object-level features (pT, η, φ). Events that reconstruct poorly are flagged as anomalous.

### Axis 2 — Contrastive encoder (clustering)

A Linformer-based Transformer encodes the PF candidates into a low-dimensional latent vector. Training uses:

- **Supervised Contrastive Loss (SupCon)** — clusters events by class in latent space
- **Cross-Entropy Loss** — sharpens class boundaries via a joint classification head

The anomaly score on this axis is the squared Mahalanobis distance of an event's latent vector from the QCD cluster, computed by fitting PCA on the QCD events in each mini-batch and measuring the squared Euclidean distance in the whitened space.

### DisCo decorrelation

For ABCD to work the two axes need to be statistically independent in the background. DisCo enforces this by penalising dependence between the AE reco loss and the Mahalanobis distance on QCD events:

- **DisCo loss** — minimises distance correlation between the two axes
- **Closure loss** — directly penalises ABCD nonclosure

---

## Setup

```bash
pip install -e .
pip install -r nrp/requirements_nrp.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.3.1.

---

## Training

```bash
python train_pca_per_batch.py \
    --train_cfg configs/train_sup_con_ce.yaml \
    --data_cfg configs/data_smcocktail.yaml \
    --data /axovol/contrastive_axis/data/hlt_smcocktail_train.pt
```

### Key config options

```yaml
hyperparameters:
  contrast_loss: SupConLoss
  contrastive_weight: 0.05    # balance between contrastive and CE loss
  disco_weight: 0.1           # decorrelation strength
  closure_weight: 0.1         # ABCD closure penalty
  disco_warmup: 0.2           # fraction of steps to ramp disco weight from zero
  pca_components: 6           # PCA dims for Mahalanobis distance
  ae_latent: 16               # AE bottleneck size
  latent_dim: 6               # encoder output dimension
```

---

## Running on NRP-Nautilus

```bash
kubectl apply -f nrp/train_job_mar25.yaml
```

Jobs mount the `/axovol` PVC for both code and data. Checkpoints go to `checkpoints/`, logs to `logs/`. Runs are tracked with W&B.

---

## Evaluation

```bash
python eval_abcd.py \
    --contrast_ckpt checkpoints/<run>.pth \
    --contrast_test_pt /axovol/contrastive_axis/data/hlt_smcocktail_test.pt \
    --signal_pt /axovol/contrastive_axis/data/signal_pt/hlt_signal_TpTp.pt \
    --outdir /axovol/contrastive_axis/abcd_outputs/<run> \
    --n_pca 6
```

import torch
import torch.nn as nn
import datetime
import os
import logging
import argparse
import importlib
import wandb
import numpy as np
from embedding_pca_epoch.models import TransformerEncoder, Projector
from embedding_pca_epoch.autoencoder import Autoencoder
from embedding_pca_epoch.training import (
    make_train_val_split, build_train_val_loaders,
    train_epoch_single_disco, validate_epoch, EarlyStopping,
    cosine_schedule_with_warmup, cosine_constrastive_schedule,
    linear_warmup_weight, fit_qcd_pca,
)
from embedding_pca_epoch.utils.data_utils import compute_normalization_constants
from embedding_pca_epoch.utils.cfg_handler import train_config, data_config
from embedding_pca_epoch.utils.data_utils import compute_class_weights

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JEPA")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training_single_disco_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)


def _load_ae_from_checkpoint(ckpt, device):
    """Infer AE architecture from checkpoint state dict and load weights."""
    ae_sd = ckpt["ae"]
    backbone_linear_keys = sorted(
        k for k in ae_sd if k.startswith("encoder.backbone.") and k.endswith(".weight")
    )
    features   = int(ae_sd[backbone_linear_keys[0]].shape[1])
    enc_nodes  = [int(ae_sd[k].shape[0]) for k in backbone_linear_keys]
    latent_dim = int(ae_sd["encoder.fc_latent.weight"].shape[0])
    dec_linear_keys = sorted(
        k for k in ae_sd
        if k.startswith("decoder.net.") and k.endswith(".weight") and ae_sd[k].ndim == 2
    )
    dec_nodes = [int(ae_sd[k].shape[0]) for k in dec_linear_keys]

    ae = Autoencoder({
        "features":       features,
        "latent_dim":     latent_dim,
        "encoder_config": {"nodes": enc_nodes},
        "decoder_config": {"nodes": dec_nodes},
        "alpha": 1.0,
    }).to(device)
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded frozen AE: input={features}, enc={enc_nodes}, latent={latent_dim}, dec={dec_nodes}")
    return ae


def _train_ae_from_scratch(obj_tr_norm, cfg, device):
    """Train a fresh AE on normalised training obj features, then freeze and return it."""
    features     = obj_tr_norm.shape[1]
    ae_latent    = cfg.hp("ae_latent", 16)
    ae_enc_nodes = cfg.hp("ae_enc_nodes", [512, 256])
    ae_dec_nodes = cfg.hp("ae_dec_nodes", [256, 512])
    ae_lr        = cfg.hp("ae_lr", 1e-3)
    ae_epochs    = cfg.hp("ae_pretrain_epochs", 20)
    ae_bs        = cfg.hp("batch_size", 4096)

    ae = Autoencoder({
        "features":       features,
        "latent_dim":     ae_latent,
        "encoder_config": {"nodes": ae_enc_nodes},
        "decoder_config": {"nodes": ae_dec_nodes + [features]},
        "alpha": 1.0,
    }).to(device)

    opt = torch.optim.Adam(ae.parameters(), lr=ae_lr)
    X = obj_tr_norm.to(device)
    ae.train()
    for ep in range(ae_epochs):
        perm = torch.randperm(len(X), device=device)
        ep_losses = []
        for i0 in range(0, len(X), ae_bs):
            xb = X[perm[i0:i0 + ae_bs]]
            recon, _ = ae(xb)
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        logger.info(f"AE pre-train epoch {ep+1}/{ae_epochs}: MSE={np.mean(ep_losses):.6f}")

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    logger.info(f"Frozen AE: input={features}, enc={ae_enc_nodes}, latent={ae_latent}, dec={ae_dec_nodes}")
    return ae


def main(data_path: str, ae_ckpt_path: str | None, cfg: train_config, cfg_data: data_config, test_mode: bool = False):

    # ── WandB ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="embedding_hlt_single_disco",
        config={**cfg.get_entire_cfg(), **cfg_data.get_entire_cfg()},
    )
    is_sweep = cfg.is_sweep()
    if not is_sweep:
        run.name = f"{cfg.get_model_name()}_single_disco_{timestamp}"
        logger.info(f"Run name: {run.name}")

    # ── Hyperparameters ────────────────────────────────────────────────────────
    num_epochs          = cfg.hp("num_epochs", 400 if not is_sweep else 50)
    patience            = cfg.hp("early_stopping_patience", 100 if not is_sweep else 20)
    val_split           = cfg.get_trdata_cfg("val_split", 0.1)
    pairwise            = cfg.get_trdata_cfg("pairwise", False)
    class_weights_setting = cfg_data.get("class_weights", None)
    pfcands             = cfg_data.get("pfcands", True)
    contrast_loss       = cfg.hp("contrast_loss", "InfoNCELoss")
    preproc_type        = cfg.get_trdata_cfg("preproc_type", "PFPreProcessor")
    mixed_prec          = cfg.hp("mixed_prec", False)

    num_heads       = cfg.hp("num_heads", 8)
    num_layers      = cfg.hp("num_layers", 4)
    embed_size      = cfg.hp("embed_size", 128)
    latent_dim      = cfg.hp("latent_dim", 6)
    proj_dim        = cfg.hp("proj_dim", 12)
    linear_dim      = cfg.hp("linear_dim", None)
    dim_feedforward = cfg.hp("dim_feedforward", 2048)
    contrast_temp   = cfg.hp("contrast_temp", 0.07)
    contrastive_weight  = cfg.hp("contrastive_weight", 1.0)
    contrastive_max     = cfg.hp("contrastive_max", None)
    contrastive_warmup  = cfg.hp("contrastive_warmup", 0.05)
    lr              = cfg.hp("lr", 1e-3)
    lr_min          = cfg.hp("lr_min", 0.0)
    lr_warmup       = cfg.hp("lr_warmup", 0.05)
    batch_size      = cfg.hp("batch_size", 256)
    use_l2_proxy_md = cfg.hp("use_l2_proxy_md", False)
    disco_weight    = cfg.hp("disco_weight", 1.0)
    closure_weight  = cfg.hp("closure_weight", 0.0)
    disco_warmup    = cfg.hp("disco_warmup", 0.0)
    closure_warmup  = cfg.hp("closure_warmup", 0.0)
    pca_n           = cfg.hp("pca_components", None)

    scaler = torch.cuda.amp.GradScaler(enabled=((device == "cuda") and mixed_prec))

    # ── Load or prepare AE ────────────────────────────────────────────────────
    if ae_ckpt_path is not None:
        logger.info(f"Loading AE from: {ae_ckpt_path}")
        ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
        ae_model = _load_ae_from_checkpoint(ae_ckpt, device)
        obj_scaler = ae_ckpt["ae_scaler"]
        del ae_ckpt
    else:
        logger.info("No AE checkpoint provided — will train AE from scratch after data loading.")
        ae_model = None
        obj_scaler = None

    # ── Data loading ───────────────────────────────────────────────────────────
    if test_mode:
        num_events = int(0.10 * cfg_data.get("nevents_per_class") * cfg_data.get_file_label_map().__len__())
        logger.info(f"Test mode: using {num_events} events")

    _raw = torch.load(data_path, map_location="cpu")
    _n = num_events if test_mode else -1
    if isinstance(_raw, dict):
        feature_block = _raw['pf'][:_n] if _n > 0 else _raw['pf']
        label_block   = (_raw['label'][:_n] if _n > 0 else _raw['label']).long()
        feature_block = torch.nan_to_num(feature_block, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        from embedding_pca_epoch.utils.data_utils import clean_data
        data = _raw[:_n] if _n > 0 else _raw
        feature_block, label_block = clean_data(data)

    # obj features — normalised using the AE checkpoint's scaler (not recomputed)
    _obj_raw = _raw["obj"][:_n] if (isinstance(_raw, dict) and "obj" in _raw and _n > 0) \
               else (_raw["obj"] if (isinstance(_raw, dict) and "obj" in _raw) else None)
    del _raw

    X_tr, y_tr, X_val, y_val, idx_tr, idx_val = make_train_val_split(feature_block, label_block, val_size=val_split)
    num_tokens_pf = feature_block.size(1)
    num_classes   = int(label_block.max().item()) + 1
    class_weights = compute_class_weights(label_block, setting=class_weights_setting)
    del feature_block, label_block

    # Normalise obj features; fit scaler from training split if training AE from scratch
    obj_tr = obj_val = None
    if _obj_raw is not None:
        if obj_scaler is None:
            obj_tr_raw = _obj_raw[idx_tr, :, :4].reshape(idx_tr.shape[0], -1).float().numpy()
            mu_np  = obj_tr_raw.mean(axis=0).astype(np.float32)
            std_np = obj_tr_raw.std(axis=0).astype(np.float32)
            std_np = np.where(std_np < 1e-8, 1.0, std_np)
            obj_scaler = {"mu": torch.from_numpy(mu_np), "std": torch.from_numpy(std_np)}
            del obj_tr_raw
            logger.info("Computed AE scaler from training obj features.")
        mu  = obj_scaler["mu"].cpu().numpy()
        std = obj_scaler["std"].cpu().numpy()
        obj_flat = _obj_raw[:, :, :4].reshape(_obj_raw.shape[0], -1).float().numpy()
        del _obj_raw
        obj_norm = torch.from_numpy(((obj_flat - mu) / (std + 1e-8)).astype(np.float32))
        del obj_flat
        obj_tr  = obj_norm[idx_tr]
        obj_val = obj_norm[idx_val]
        del obj_norm

    # Train AE from scratch if no checkpoint was provided
    if ae_model is None:
        if obj_tr is None:
            raise RuntimeError("Cannot train AE from scratch: no obj features found in data.")
        logger.info("Training AE from scratch...")
        ae_model = _train_ae_from_scratch(obj_tr, cfg, device)

    # ── Class balance logging ──────────────────────────────────────────────────
    for split_name, y_split in [("train", y_tr), ("val", y_val)]:
        counts = torch.bincount(y_split, minlength=num_classes)
        logger.info(f"Class counts ({split_name}): " + ", ".join(f"{i}:{counts[i].item()}" for i in range(num_classes)))

    # ── DataLoaders ────────────────────────────────────────────────────────────
    norm_constants = compute_normalization_constants(X_tr) if not pfcands else {}
    train_loader, val_loader = build_train_val_loaders(
        X_tr, y_tr, X_val, y_val, device=device, batch_size=batch_size, pfcands=pfcands,
        obj_tr=obj_tr, obj_val=obj_val,
    )

    # ── Model construction ─────────────────────────────────────────────────────
    preproc_class = getattr(importlib.import_module("embedding.preprocs"), preproc_type)
    preproc = preproc_class(norm_constants).to(device)

    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=embed_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        linear_dim=linear_dim,
        num_tokens=num_tokens_pf if linear_dim is not None else None,
        pairwise=pairwise,
        pre_processor=preproc,
    ).to(device).train()
    projector  = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim * 4)).to(device).train()
    classifier = nn.Linear(proj_dim, num_classes).to(device).train()

    # ── Loss functions ─────────────────────────────────────────────────────────
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    contrast_loss_class = getattr(importlib.import_module("embedding.loss"), contrast_loss)
    criterion = contrast_loss_class(temperature=contrast_temp)

    # ── Optimizer (contrastive only — AE is frozen) ───────────────────────────
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(projector.parameters()) +
        list(classifier.parameters()),
        lr=lr,
    )

    # ── Schedulers ─────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = num_epochs * steps_per_epoch
    warmup_steps    = int(lr_warmup * total_steps)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, lr=lr, lr_min=lr_min)

    if contrastive_max is not None:
        contrastive_schedule = cosine_constrastive_schedule(
            weight_min=contrastive_weight, weight_max=contrastive_max,
            warmup_steps=int(contrastive_warmup * total_steps), total_steps=total_steps,
        )

    disco_schedule   = linear_warmup_weight(disco_weight,   int(disco_warmup   * total_steps)) if disco_warmup   > 0.0 and disco_weight   > 0.0 else disco_weight
    closure_schedule = linear_warmup_weight(closure_weight, int(closure_warmup * total_steps)) if closure_warmup > 0.0 and closure_weight > 0.0 else closure_weight

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val = float("inf")
    es = EarlyStopping(patience=patience, mode="min", min_delta=0.0)
    model_path = os.path.join(os.getcwd(), "checkpoints", f"{cfg.get_model_name()}_single_disco_encoder_{timestamp}.pth")

    use_pca = (pca_n is not None) and (disco_weight > 0.0 or closure_weight > 0.0)
    pca_mean = pca_comps = pca_stds = None

    logger.info(f"Starting single DisCo training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        if use_pca:
            pca_mean, pca_comps, pca_stds, explained = fit_qcd_pca(
                encoder, projector, train_loader, norm_constants, pairwise, device,
                n_components=pca_n, use_l2_proxy_md=use_l2_proxy_md,
            )
            logger.info(f"Epoch {epoch+1}: PCA fitted ({pca_n} components, {100*explained:.1f}% variance)")

        tr = train_epoch_single_disco(
            encoder, projector, classifier,
            ce_loss_fn, criterion,
            train_loader, norm_constants, device,
            optimizer, scheduler=scheduler,
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, num_classes=num_classes, scaler=scaler,
            ae_model=ae_model,
            disco_weight=disco_schedule, closure_weight=closure_schedule,
            use_l2_proxy_md=use_l2_proxy_md, pca_mean=pca_mean, pca_comps=pca_comps, pca_stds=pca_stds,
        )
        va = validate_epoch(
            encoder, projector, classifier,
            ce_loss_fn, criterion,
            val_loader, norm_constants, device,
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise, num_classes=num_classes,
            ae_model=None, ae_reco_weight=0.0,  # AE not in val loss
        )

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: loss {tr['loss']:.6f}, Contrast {tr['contrast']:.6f}, "
            f"CE {tr['ce']:.6f}, DisCo {tr['disco']:.6f}, Closure {tr['closure']:.6f}, acc {tr['acc']:.4f} | "
            f"Val: loss {va['loss']:.6f}, Contrast {va['contrast']:.6f}, CE {va['ce']:.6f}, acc {va['acc']:.4f}"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "encoder":    encoder.state_dict(),
                "projector":  projector.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "norm_constants": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_constants.items()},
                # include frozen AE so eval_abcd.py can load it from this checkpoint
                "ae":         ae_model.state_dict(),
                "ae_scaler":  obj_scaler,
            }
            if use_pca and pca_mean is not None:
                ckpt["pca_mean"]        = pca_mean.cpu()
                ckpt["pca_components"]  = pca_comps.cpu()
                ckpt["pca_stds"]        = pca_stds.cpu()
                ckpt["pca_n_components"] = pca_n
            torch.save(ckpt, model_path)
            logger.info(f"Saved best checkpoint: {model_path}")

        run.log({
            "Train Loss":        tr["loss"],
            "Train Contrastive": tr["contrast"],
            "Train CrossEntropy": tr["ce"],
            "Train DisCo":       tr["disco"],
            "Train Closure":     tr["closure"],
            "Train Accuracy":    tr["acc"],
            "Val Loss":          va["loss"],
            "Val Contrastive":   va["contrast"],
            "Val CrossEntropy":  va["ce"],
            "Val Accuracy":      va["acc"],
            "Learning Rate":     scheduler.get_last_lr()[0],
            "Contrastive Weight": contrastive_weight if contrastive_max is None else contrastive_schedule.get(),
            "DisCo Weight":      disco_schedule if isinstance(disco_schedule, float) else disco_schedule.get(),
            "Closure Weight":    closure_schedule if isinstance(closure_schedule, float) else closure_schedule.get(),
        }, step=epoch)

        if es.step(va["loss"]):
            logger.info("Early stopping triggered.")
            break

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg",   required=True, help="Path to data config .yaml")
    parser.add_argument("--train_cfg",  required=True, help="Path to training config .yaml")
    parser.add_argument("--data",       required=True, help="Path to input .pt file")
    parser.add_argument("--ae_ckpt",    default=None, help="Path to checkpoint with pre-trained AE (optional; trains from scratch if omitted)")
    parser.add_argument("--test_mode",  action="store_true")
    args = parser.parse_args()

    tr_cfg   = train_config(args.train_cfg)
    data_cfg = data_config(args.data_cfg)

    logger.info(f"Train config: {tr_cfg.get_entire_cfg()}")
    logger.info(f"Data config:  {data_cfg.get_entire_cfg()}")

    main(args.data, args.ae_ckpt, tr_cfg, data_cfg, test_mode=args.test_mode)

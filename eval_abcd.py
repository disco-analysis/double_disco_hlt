import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic

from embedding.models import TransformerEncoder, Projector
from embedding.autoencoder import Autoencoder
from embedding.preprocs import PFPreProcessor
from embedding.utils.data_utils import load_data


# ─────────────────────────────────────────────
# ABCD helpers
# ─────────────────────────────────────────────

def abcd_counts(loss_1, loss_2, percent_1, percent_2):
    thresh_1 = np.quantile(loss_1, percent_1)
    thresh_2 = np.quantile(loss_2, percent_2)
    A = int(((loss_1 > thresh_1) & (loss_2 > thresh_2)).sum())
    B = int(((loss_1 > thresh_1) & (loss_2 <= thresh_2)).sum())
    C = int(((loss_1 <= thresh_1) & (loss_2 > thresh_2)).sum())
    D = int(((loss_1 <= thresh_1) & (loss_2 <= thresh_2)).sum())
    return thresh_1, thresh_2, A, B, C, D


def nonclosure_A(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A_hat <= 0:
        return np.inf, A_hat
    return (A - A_hat) / A_hat, A_hat


def profile_plot(ax, x, y, nbins=30, logx=False, min_per_bin=20, label="mean ± SE"):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    x, y = x[m], y[m]

    xu = np.log10(x) if logx else x
    lo, hi = float(xu.min()), float(xu.max())
    if lo == hi:
        hi = np.nextafter(hi, np.inf)
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean, _, _ = binned_statistic(xu, y, statistic="mean",  bins=edges)
    std,  _, _ = binned_statistic(xu, y, statistic="std",   bins=edges)
    cnt,  _, _ = binned_statistic(xu, y, statistic="count", bins=edges)
    sem = std / np.sqrt(np.maximum(cnt, 1))

    good = cnt >= min_per_bin
    xc = centers[good]
    xplot = (10.0 ** xc) if logx else xc
    if logx:
        ax.set_xscale("log")

    ax.errorbar(xplot, mean[good], yerr=sem[good],
                fmt="o", ms=3, lw=1, capsize=2, label=label)
    ax.grid(alpha=0.3)
    return {"x": xplot, "mean": mean[good], "sem": sem[good], "count": cnt[good]}


# ─────────────────────────────────────────────
# AE inference from joint checkpoint
# ─────────────────────────────────────────────

def compute_ae_scores(ckpt, test_pt_path, device="cpu", batch_size=4096):
    """
    Loads the AE stored in a joint checkpoint, runs inference on the obj features
    from test_pt_path, and returns per-event MSE reco loss.
    """
    ae_sd = ckpt["ae"]

    # infer architecture from state dict
    backbone_linear_keys = sorted(
        k for k in ae_sd if k.startswith("encoder.backbone.") and k.endswith(".weight")
    )
    features  = int(ae_sd[backbone_linear_keys[0]].shape[1])
    enc_nodes = [int(ae_sd[k].shape[0]) for k in backbone_linear_keys]
    latent_dim = int(ae_sd["encoder.fc_latent.weight"].shape[0])
    dec_linear_keys = sorted(
        k for k in ae_sd if k.startswith("decoder.net.") and k.endswith(".weight") and ae_sd[k].ndim == 2
    )
    dec_nodes = [int(ae_sd[k].shape[0]) for k in dec_linear_keys]

    ae = Autoencoder({
        "features":       features,
        "latent_dim":     latent_dim,
        "encoder_config": {"nodes": enc_nodes},
        "decoder_config": {"nodes": dec_nodes},
        "alpha": 1.0,
    }).to(device).eval()
    ae.load_state_dict(ae_sd)
    print(f"  AE: input={features}, enc={enc_nodes}, latent={latent_dim}, dec={dec_nodes}", flush=True)

    # load and normalise obj features the same way as during training
    scaler = ckpt["ae_scaler"]
    mu  = scaler["mu"].cpu().numpy()
    std = scaler["std"].cpu().numpy()

    data = torch.load(test_pt_path, map_location="cpu")
    obj = data["obj"][:, :, :4].reshape(data["obj"].shape[0], -1).float().numpy()
    obj_norm = torch.from_numpy(((obj - mu) / (std + 1e-8)).astype(np.float32))
    N = obj_norm.shape[0]

    print(f"  Running AE inference on {N} events...", flush=True)
    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = obj_norm[i0:i0 + batch_size].to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    return torch.cat(scores).numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Contrastive model: embed + Mahalanobis
# ─────────────────────────────────────────────

def compute_md_scores(ckpt_path, test_pt_path, device="cpu", batch_size=4096):
    """
    Runs encoder+projector on test_pt_path, fits a Mahalanobis reference
    on QCD (label==1) embeddings, and returns per-event MD scores for all events.
    """
    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_sd  = ckpt["encoder"]
    proj_sd = ckpt["projector"]

    # infer architecture from checkpoint
    embed_size = int(enc_sd["cls_token"].shape[-1]) if "cls_token" in enc_sd \
        else int(enc_sd["input_proj.weight"].shape[0])
    latent_dim = int(enc_sd["bottleneck.weight"].shape[0])

    layer_ids = [int(k.split(".")[1]) for k in enc_sd if k.startswith("layers.")]
    num_layers = (max(layer_ids) + 1) if layer_ids else 0

    num_heads = next(
        (int(v.shape[0]) for k, v in enc_sd.items()
         if k.endswith("self_attn.bias_mlp.2.bias")),
        4
    )
    pairwise = ckpt.get("pairwise", False)

    # detect Linformer (LinearAttentionLayer) from e_proj/f_proj keys
    linformer = any("self_attn.e_proj" in k for k in enc_sd)
    if linformer:
        e_proj_key = next(k for k in enc_sd if k.endswith("self_attn.e_proj.weight"))
        w = enc_sd[e_proj_key]
        linear_dim  = int(w.shape[0])
        num_tokens  = int(w.shape[1]) - 1  # TransformerEncoder adds 1 for CLS token internally
    else:
        linear_dim = None
        num_tokens = None

    linear_w = sorted([(k, v) for k, v in proj_sd.items()
                        if hasattr(v, "ndim") and v.ndim == 2],
                       key=lambda kv: kv[0])
    proj_dim = int(linear_w[-1][1].shape[0])

    norm_constants = ckpt.get("norm_constants", {})
    preproc = PFPreProcessor(norm_constants).to(device)

    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=embed_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=linear_dim,
        num_tokens=num_tokens,
        pairwise=pairwise,
        pre_processor=preproc,
    ).to(device).eval()

    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim * 4)).to(device).eval()

    encoder.load_state_dict(enc_sd)
    projector.load_state_dict(proj_sd)

    print(f"  embed_size={embed_size}, latent_dim={latent_dim}, "
          f"num_heads={num_heads}, num_layers={num_layers}, pairwise={pairwise}", flush=True)

    # load test data
    data = torch.load(test_pt_path, map_location="cpu")
    pf     = data["pf"]      # [N, 400, 7]
    labels = data["label"]   # [N]
    N = pf.shape[0]

    print(f"  Running inference on {N} events...", flush=True)
    embeddings = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            mask = (xb.abs().sum(-1) == 0)  # [B, T] padding mask
            mask = torch.cat([
                torch.zeros(mask.size(0), 1, device=device, dtype=torch.bool),
                mask
            ], dim=1)
            latent = encoder(xb, None, mask)
            z = F.normalize(projector(latent), dim=1)
            embeddings.append(z.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()  # [N, proj_dim]

    # fit reference on QCD (label==1)
    qcd_mask = (labels.numpy() == 1)
    ref = embeddings[qcd_mask]
    print(f"  Fitting Mahalanobis on {qcd_mask.sum()} QCD events...", flush=True)

    mu = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False) + 1e-6 * np.eye(ref.shape[1])
    inv_cov = np.linalg.inv(cov)

    diffs = embeddings - mu
    md = np.einsum("bi,ij,bj->b", diffs, inv_cov, diffs).astype(np.float32)
    return md, labels.numpy()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def ABCD(config):
    print("Logging in to wandb...", flush=True)
    wandb.login()
    wandb.init(project="AE vs. Contrastive ABCD",
               settings=wandb.Settings(_disable_stats=True),
               config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}", flush=True)

    outdir   = config.get("outdir", "outputs_abcd")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load checkpoint once (shared by both AE and contrastive inference) ──
    ckpt = torch.load(config["contrast_ckpt"], map_location=device)

    # ── AE scores: compute from joint checkpoint if available, else load pre-computed ──
    if "ae" in ckpt and "ae_scaler" in ckpt:
        print("Computing AE scores from joint checkpoint...", flush=True)
        ae_bkg = compute_ae_scores(ckpt, config["contrast_test_pt"], device=device)
    elif config.get("ae_scores_bkg_test"):
        print("Loading pre-computed AE scores...", flush=True)
        ae_bkg = torch.load(config["ae_scores_bkg_test"], map_location="cpu").numpy().astype(np.float32).reshape(-1)
    else:
        raise ValueError("No AE scores available: checkpoint has no 'ae' key and --ae_scores_bkg_test not provided.")

    # ── compute contrastive MD scores ──
    con_bkg, labels = compute_md_scores(
        config["contrast_ckpt"],
        config["contrast_test_pt"],
        device=device,
    )

    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Length mismatch: contrastive {len(con_bkg)} vs AE {len(ae_bkg)}")

    # ── mask (finite, positive AE loss) ──
    mask = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)
    axis1_bkg = ae_bkg[mask]    # AE reco loss  → x axis
    axis2_bkg = con_bkg[mask]   # contrastive MD → y axis
    labels_masked = labels[mask]
    print(f"Events after masking: {mask.sum()}", flush=True)

    # ── ABCD scan ──
    percent = np.linspace(0.75, 0.98, 24)
    best    = {"nonclosure": np.inf}
    min_A   = int(config.get("min_A", 200))
    min_D   = int(config.get("min_D", 1000))

    for p1 in percent:
        for p2 in percent:
            t1, t2, A, B, C, D = abcd_counts(axis1_bkg, axis2_bkg, p1, p2)
            if A < min_A or D < min_D:
                continue
            nc, A_hat = nonclosure_A(A, B, C, D)
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update(dict(p1=p1, p2=p2, t1=t1, t2=t2,
                                 A=A, B=B, C=C, D=D, A_hat=A_hat, nonclosure=nc))

    if "t1" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering min_A/min_D.")

    t1_opt, t2_opt = best["t1"], best["t2"]
    print(f"Optimized: p1={best['p1']:.3f}, p2={best['p2']:.3f}", flush=True)
    print(f"Thresholds: t1={t1_opt:.4g}, t2={t2_opt:.4g}", flush=True)
    print(f"Nonclosure: {100.0*best['nonclosure']:.2f}%", flush=True)

    wandb.log({
        "ABCD/opt_p1":      best["p1"],
        "ABCD/opt_p2":      best["p2"],
        "ABCD/opt_t1":      float(t1_opt),
        "ABCD/opt_t2":      float(t2_opt),
        "ABCD/nonclosure":  float(best["nonclosure"]),
        "ABCD/A": int(best["A"]), "ABCD/B": int(best["B"]),
        "ABCD/C": int(best["C"]), "ABCD/D": int(best["D"]),
    })

    # ─────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────
    fs, fs_leg = 28, 24
    fig_size   = (8, 6)

    # 2D histogram
    fig = plt.figure(figsize=(6, 5))
    xbins = np.geomspace(axis1_bkg[axis1_bkg > 0].min(), axis1_bkg.max(), 201)
    ybins = np.geomspace(axis2_bkg[axis2_bkg > 0].min(), axis2_bkg.max(), 201)
    plt.hist2d(axis1_bkg, axis2_bkg, bins=[xbins, ybins], norm=LogNorm(vmin=1), cmin=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("AE reco loss")
    plt.ylabel("Contrastive score (MD)")
    plt.title("AE vs Contrastive (bkg only)")
    plt.colorbar(label="Counts")
    out = os.path.join(plot_dir, "hist2d_bkg.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Hists2D/bkg": wandb.Image(out)})

    # per-class scatter + individual hist2d
    class_names  = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    class_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

    # combined scatter coloured by class
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() == 0:
            continue
        ax.scatter(axis1_bkg[m], axis2_bkg[m], s=0.3, alpha=0.15,
                   color=class_colors[cls], label=name, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AE reco loss", fontsize=fs)
    ax.set_ylabel("Contrastive score (MD)", fontsize=fs)
    ax.set_title("AE vs Contrastive — all classes")
    ax.legend(markerscale=10, fontsize=fs_leg)
    out_combined = os.path.join(plot_dir, "hist2d_by_class_combined.png")
    fig.savefig(out_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Hists2D/by_class_combined": wandb.Image(out_combined)})

    # individual hist2d per class
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 2:
            continue
        x_cls = axis1_bkg[m]
        y_cls = axis2_bkg[m]
        fig = plt.figure(figsize=(6, 5))
        xbins_c = np.geomspace(x_cls[x_cls > 0].min(), x_cls.max(), 101)
        ybins_c = np.geomspace(y_cls[y_cls > 0].min(), y_cls.max(), 101)
        plt.hist2d(x_cls, y_cls, bins=[xbins_c, ybins_c], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel("Contrastive score (MD)", fontsize=fs)
        plt.title(f"AE vs Contrastive — {name}")
        plt.colorbar(label="Counts")
        out_cls = os.path.join(plot_dir, f"hist2d_{name}.png")
        plt.savefig(out_cls, dpi=200, bbox_inches="tight")
        plt.close()
        wandb.log({f"Hists2D/{name}": wandb.Image(out_cls)})

    # profile: <AE> vs contrastive
    fig, ax = plt.subplots(figsize=fig_size)
    profile_plot(ax, axis2_bkg, axis1_bkg, nbins=60, logx=True)
    ax.set_xlabel("Contrastive score (MD)", fontsize=fs)
    ax.set_ylabel("Mean AE reco loss",      fontsize=fs)
    ax.set_yscale("log")
    ax.set_title("⟨AE loss⟩ vs contrastive MD")
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p1_path = os.path.join(plot_dir, "profile_AE_vs_contrastive.png")
    fig.savefig(p1_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/AE_vs_contrastive": wandb.Image(p1_path)})

    # profile: <contrastive> vs AE
    fig, ax = plt.subplots(figsize=fig_size)
    profile_plot(ax, axis1_bkg, axis2_bkg, nbins=60, logx=True)
    ax.set_xlabel("AE reco loss",                  fontsize=fs)
    ax.set_ylabel("Mean contrastive score (MD)",   fontsize=fs)
    ax.set_title("⟨contrastive MD⟩ vs AE loss")
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p2_path = os.path.join(plot_dir, "profile_contrastive_vs_AE.png")
    fig.savefig(p2_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/contrastive_vs_AE": wandb.Image(p2_path)})

    # per-class profile: <AE> vs contrastive MD
    fig, ax = plt.subplots(figsize=fig_size)
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 20:
            continue
        profile_plot(ax, axis2_bkg[m], axis1_bkg[m], nbins=40, logx=True, label=name)
    ax.set_xlabel("Contrastive score (MD)", fontsize=fs)
    ax.set_ylabel("Mean AE reco loss",      fontsize=fs)
    ax.set_yscale("log")
    ax.set_title("⟨AE loss⟩ vs contrastive MD (by class)")
    ax.legend(fontsize=fs_leg)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p3_path = os.path.join(plot_dir, "profile_AE_vs_contrastive_by_class.png")
    fig.savefig(p3_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/AE_vs_contrastive_by_class": wandb.Image(p3_path)})

    # per-class profile: <contrastive MD> vs AE
    fig, ax = plt.subplots(figsize=fig_size)
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 20:
            continue
        profile_plot(ax, axis1_bkg[m], axis2_bkg[m], nbins=40, logx=True, label=name)
    ax.set_xlabel("AE reco loss",                  fontsize=fs)
    ax.set_ylabel("Mean contrastive score (MD)",   fontsize=fs)
    ax.set_title("⟨contrastive MD⟩ vs AE loss (by class)")
    ax.legend(fontsize=fs_leg)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p4_path = os.path.join(plot_dir, "profile_contrastive_vs_AE_by_class.png")
    fig.savefig(p4_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/contrastive_vs_AE_by_class": wandb.Image(p4_path)})

    # 1D scan for closure + S/sqrt(B)
    effs, closure_ratio, closure_unc, s_over_sqrtb = [], [], [], []
    Ntot_bkg = float(len(axis1_bkg))

    for p in percent:
        t1, t2, A, B, C, D = abcd_counts(axis1_bkg, axis2_bkg, p, p)
        A_hat  = (B * C) / max(D, 1e-8)
        ratio  = A_hat / max(A, 1e-8)
        invA   = 0.0 if A == 0 else 1.0 / A
        invB   = 0.0 if B == 0 else 1.0 / B
        invC   = 0.0 if C == 0 else 1.0 / C
        invD   = 0.0 if D == 0 else 1.0 / D
        rel_var = invA + invB + invC + invD
        sigma  = abs(ratio) * np.sqrt(rel_var) if rel_var > 0 else 0.0

        effs.append(A / max(Ntot_bkg, 1.0))
        closure_ratio.append(ratio)
        closure_unc.append(sigma)
        s_over_sqrtb.append(0.0 / np.sqrt(max(A, 1e-8)))  # no signal yet

    effs          = np.array(effs)
    closure_ratio = np.array(closure_ratio)
    closure_unc   = np.array(closure_unc)
    s_over_sqrtb  = np.array(s_over_sqrtb)

    order         = np.argsort(effs)
    effs          = effs[order]
    closure_ratio = closure_ratio[order]
    closure_unc   = closure_unc[order]
    s_over_sqrtb  = s_over_sqrtb[order]

    eff_opt   = best["A"] / max(Ntot_bkg, 1.0)
    ratio_opt = best["A_hat"] / max(best["A"], 1e-8)

    # closure plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, closure_ratio, c="g", label="AE + Contrastive (MD)")
    ax.fill_between(effs,
                    closure_ratio - closure_unc,
                    closure_ratio + closure_unc,
                    facecolor="g", alpha=0.5, interpolate=True)
    ax.plot(effs, np.ones_like(effs),          linestyle="-",  color="black")
    ax.plot(effs, np.full_like(effs, 0.95),    linestyle="--", color="black")
    ax.plot(effs, np.full_like(effs, 1.05),    linestyle="--", color="black")
    ax.plot([eff_opt], [ratio_opt], marker="o", c="red", label="Optimized")
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel("Predicted Bkg. / True Bkg.",        fontsize=fs)
    ax.set_ylim([0.0, 1.5])
    ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="lower right", fontsize=fs_leg)
    closure_path = os.path.join(plot_dir, "cut_and_count_bkg_check.png")
    plt.savefig(closure_path, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Closure/plot": wandb.Image(closure_path)})

    # S/sqrt(B) placeholder
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, s_over_sqrtb, color="red", label=r"$S/\sqrt{B}$")
    ax.plot([eff_opt], [0.0], marker="o", color="black")
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel(r"$S/\sqrt{B}$",                     fontsize=fs)
    ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="best", fontsize=fs_leg)
    sig_path = os.path.join(plot_dir, "s_over_sqrtb_vs_bkg_eff.png")
    plt.savefig(sig_path, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Signal/s_over_sqrtb_vs_bkg_eff": wandb.Image(sig_path)})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast_ckpt",      required=True)
    parser.add_argument("--contrast_test_pt",   required=True)
    parser.add_argument("--ae_scores_bkg_test", default=None,
                        help="Pre-computed AE scores .pt file. Not needed if checkpoint contains a joint AE.")
    parser.add_argument("--outdir",             default="outputs_abcd")
    parser.add_argument("--min_A", type=int,    default=200)
    parser.add_argument("--min_D", type=int,    default=1000)
    args = parser.parse_args()

    ABCD(vars(args))

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_seq import CSISeqDataset, load_norm_stats, variant_files
from metrics import (compute_all_metrics, iou_score, dice_coefficient,
                     pixel_accuracy)
from model_stage3 import build_model


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="/mnt/user-data/uploads")
    p.add_argument("--run_dir",    type=str, required=True)
    p.add_argument("--split",      type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--n_examples", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers",type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Temporal Consistency (3.6.3) — Variant B only
# ─────────────────────────────────────────────────────────────

def temporal_consistency(pred_bin_seq: torch.Tensor) -> torch.Tensor:
    """
    pred_bin_seq : (N, T, W, W) binarized predictions.
    TC = mean over samples of (1/(T-1)) Σ_t mean|Ŷ_t − Ŷ_{t-1}|.
    Lower = smoother motion. Returns a scalar tensor.
    """
    if pred_bin_seq.shape[1] < 2:
        return torch.tensor(0.0)
    diff = (pred_bin_seq[:, 1:] - pred_bin_seq[:, :-1]).abs()   # (N,T-1,W,W)
    return diff.mean()


# ─────────────────────────────────────────────────────────────
# Inference over a split
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (probas, targets) on CPU, original (un-flattened) shapes."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu()
        P.append(p)
        Y.append(y)
    return torch.cat(P, 0), torch.cat(Y, 0)


def per_sample_iou(y_true, proba, threshold):
    """Per-sample IoU on (N,W,W) tensors."""
    yb = (proba >= threshold).float()
    inter = (yb * y_true).sum(dim=(-2, -1))
    union = yb.sum(dim=(-2, -1)) + y_true.sum(dim=(-2, -1)) - inter
    return ((inter + 1e-6) / (union + 1e-6))


# ─────────────────────────────────────────────────────────────
# Qualitative grid
# ─────────────────────────────────────────────────────────────

def save_qual_grid(variant, x_norm, proba, y_true, threshold, n, path):
    """
    Variant A : rows of [CSI(last frame) | GT | Pred].
    Variant B : rows of [GT sequence (top) over Pred sequence (bottom)] for a
                few samples, showing temporal evolution.
    """

    if variant == "a":
        n = min(n, x_norm.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        if n == 1:
            axes = axes[None, :]
        for i in range(n):
            axes[i, 0].imshow(x_norm[i, -1], aspect="auto", cmap="viridis")
            axes[i, 1].imshow(y_true[i], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow((proba[i] >= threshold), cmap="gray", vmin=0, vmax=1)
            for j, t in enumerate(["CSI (frame 9)", "GT", "Pred"]):
                axes[i, j].set_title(t, fontsize=8)
                axes[i, j].axis("off")
    else:
        # Variant B: show T frames, GT row vs Pred row, for a few samples
        n = min(n, x_norm.shape[0])
        T = y_true.shape[1]
        fig, axes = plt.subplots(2 * n, T, figsize=(1.1 * T, 2.2 * n))
        if n == 1:
            axes = axes.reshape(2, T)
        for i in range(n):
            for t in range(T):
                axes[2 * i, t].imshow(y_true[i, t], cmap="gray", vmin=0, vmax=1)
                axes[2 * i + 1, t].imshow((proba[i, t] >= threshold),
                                          cmap="gray", vmin=0, vmax=1)
                axes[2 * i, t].axis("off")
                axes[2 * i + 1, t].axis("off")
                if t == 0:
                    axes[2 * i, t].set_ylabel(f"s{i} GT", fontsize=7)
                    axes[2 * i + 1, t].set_ylabel(f"s{i} Pred", fontsize=7)
            axes[2 * i, 0].set_title("t=0", fontsize=7)
    fig.suptitle(f"Stage 3 Variant {variant.upper()} — qualitative ({path.stem})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    run = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = json.load(open(run / "config.json"))
    stats = load_norm_stats(run / "norm_stats.json")
    variant = config["variant"]

    Xf, Yf = variant_files(variant)
    ds = CSISeqDataset(f"{args.data_dir}/{Xf}", f"{args.data_dir}/{Yf}",
                       f"{args.data_dir}/seq_{args.split}_idx.npy", stats,
                       in_memory=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = build_model(
        variant,
        latent_dim=config["latent_dim"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        bidirectional=config["bidirectional"],
        mask_size=config["mask_size"],
        dropout=config["dropout"],
        prev_embed_dim=config.get("prev_embed_dim", 64),
    ).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    proba, y_true = collect_predictions(model, loader, device)   # original shapes

    # Flatten time for static metrics (Variant B)
    if proba.dim() == 4:
        p_flat = proba.reshape(-1, proba.shape[-2], proba.shape[-1])
        y_flat = y_true.reshape(-1, y_true.shape[-2], y_true.shape[-1])
    else:
        p_flat, y_flat = proba, y_true

    metrics = compute_all_metrics(y_flat, p_flat, args.threshold)
    metrics["split"] = args.split
    metrics["variant"] = variant
    metrics["best_epoch"] = ckpt.get("epoch")

    if variant == "b":
        pred_bin_seq = (proba >= args.threshold).float()
        metrics["temporal_consistency"] = float(temporal_consistency(pred_bin_seq))

    with open(run / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Per-sample CSV (per-window mean IoU/Dice; for B that's frame-averaged)
    if proba.dim() == 4:
        N, T = proba.shape[0], proba.shape[1]
        ious = per_sample_iou(y_flat, p_flat, args.threshold).reshape(N, T).mean(1)
    else:
        ious = per_sample_iou(y_true, proba, args.threshold)
    with open(run / f"per_sample_metrics_{args.split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index_in_split", "iou"])
        for i, v in enumerate(ious.tolist()):
            w.writerow([i, f"{v:.4f}"])

    # Qualitative grid (sort by IoU so the grid shows a representative spread)
    order = torch.argsort(ious, descending=True)
    sel = torch.cat([order[:args.n_examples // 2], order[-(args.n_examples // 2):]])
    x_sel = torch.stack([ds[int(i)][0] for i in sel])   # (n, T, 64, 52)
    grid_path = run / f"qual_grid_{args.split}.png"
    save_qual_grid(variant, x_sel.numpy(),
                   proba[sel].numpy(), y_true[sel].numpy(),
                   args.threshold, args.n_examples, grid_path)

    # Report
    print(f"\nStage 3 Variant {variant.upper()} — {args.split} split "
          f"(best epoch {ckpt.get('epoch')})")
    print(f"  IoU       {metrics['iou']:.4f}")
    print(f"  Dice      {metrics['dice']:.4f}")
    print(f"  PixAcc    {metrics['pixel_acc']:.4f}")
    print(f"  MSE       {metrics['mse']:.5f}")
    if variant == "b":
        print(f"  TempCons  {metrics['temporal_consistency']:.5f}  (lower=smoother)")
    print(f"\nWrote metrics_{args.split}.json, "
          f"per_sample_metrics_{args.split}.csv, {grid_path.name}")


if __name__ == "__main__":
    main()

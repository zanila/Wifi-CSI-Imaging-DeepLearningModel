from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import CSIFrameDataset, load_norm_stats
from metrics import (compute_all_metrics, pixel_accuracy,
                     iou_score, dice_coefficient, mse as mse_metric)
from Stage2.model_stage2   import StaticReconstructor


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="/mnt/user-data/uploads")
    p.add_argument("--run_dir",     type=str, required=True)
    p.add_argument("--split",       type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--n_examples",  type=int,   default=12,
                   help="Number of qualitative samples in the grid")
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--num_workers", type=int,   default=2)
    return p.parse_args()


def pick_device(arg):
    if arg is not None: return torch.device(arg)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def gather_predictions(model, loader, device):
    """Run the model over the loader and return (Y, P, X) full tensors."""
    model.eval()
    all_x, all_y, all_p = [], [], []
    for x, y in loader:
        x_d = x.to(device)
        logits = model(x_d)
        proba  = torch.sigmoid(logits).cpu()
        all_x.append(x.cpu()); all_y.append(y); all_p.append(proba)
    return (torch.cat(all_x, 0), torch.cat(all_y, 0), torch.cat(all_p, 0))


def save_qual_grid(X, Y, P, threshold, path, n=12, ids=None):
    """Save a grid of (CSI, GT, prediction) triplets sampled across the split."""
    n = min(n, len(X))
    sel = np.linspace(0, len(X) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, i in enumerate(sel):
        ax_csi, ax_gt, ax_pr = axes[row]
        ax_csi.imshow(X[i].numpy(), aspect="auto", cmap="viridis")
        ax_csi.set_title(f"CSI (i={int(i)})" if ids is None
                         else f"CSI (img {int(ids[i])})", fontsize=9)
        ax_csi.set_xlabel("F"); ax_csi.set_ylabel("T")
        ax_csi.set_xticks([]);  ax_csi.set_yticks([])

        ax_gt.imshow(Y[i].numpy(), cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("GT mask", fontsize=9)
        ax_gt.set_xticks([]);   ax_gt.set_yticks([])

        bin_pred = (P[i].numpy() >= threshold).astype(np.uint8)
        ax_pr.imshow(bin_pred, cmap="gray", vmin=0, vmax=1)

        gt_t  = Y[i].unsqueeze(0); pr_t = P[i].unsqueeze(0)
        iou_v = iou_score(gt_t, pr_t, threshold).item()
        ax_pr.set_title(f"Pred (IoU={iou_v:.2f})", fontsize=9)
        ax_pr.set_xticks([]);   ax_pr.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def per_sample_metrics_csv(Y, P, threshold, path, ids=None):
    """Write a CSV with per-sample IoU/Dice/Acc/MSE for downstream analysis."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "image_id", "pixel_acc", "iou", "dice", "mse"])
        for i in range(len(Y)):
            gt = Y[i].unsqueeze(0); pr = P[i].unsqueeze(0)
            w.writerow([
                i,
                "" if ids is None else int(ids[i]),
                f"{pixel_accuracy(gt, pr, threshold).item():.5f}",
                f"{iou_score(gt, pr, threshold).item():.5f}",
                f"{dice_coefficient(gt, pr, threshold).item():.5f}",
                f"{mse_metric(gt, pr).item():.5f}",
            ])


def main():
    args = get_args()
    device = pick_device(args.device)
    data_dir = Path(args.data_dir)
    run_dir  = Path(args.run_dir)

    # ── Load checkpoint + config ──
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val IoU={ckpt['val_metrics']['iou']:.4f})")

    # ── Rebuild model with the same architecture ──
    model = StaticReconstructor(
        latent_dim=cfg["latent_dim"],
        mask_size=cfg["mask_size"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Reload normalization stats ──
    norm_stats = load_norm_stats(run_dir / "norm_stats.json")

    # ── Build the requested split ──
    idx_path = data_dir / f"frame_{args.split}_idx.npy"
    ds = CSIFrameDataset(
        data_dir / "X_frame.npy",
        data_dir / "Y_frame.npy",
        idx_path,
        norm_stats,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"))

    # ── Predict ──
    X, Y, P = gather_predictions(model, dl, device)
    print(f"Inferred {len(X)} samples on '{args.split}' split")

    overall = compute_all_metrics(Y, P, threshold=args.threshold)
    print(f"Overall metrics @ τ={args.threshold}: {overall}")

    # ── Image IDs for the chosen split (if available) ──
    ids = None
    img_ids_path = data_dir / "frame_image_ids.npy"
    if img_ids_path.exists():
        all_ids = np.load(img_ids_path, allow_pickle=True)
        split_idx = np.load(idx_path)
        ids = all_ids[split_idx]

    # ── Save outputs ──
    save_qual_grid(X, Y, P, args.threshold,
                   run_dir / f"qual_grid_{args.split}.png",
                   n=args.n_examples, ids=ids)
    per_sample_metrics_csv(Y, P, args.threshold,
                           run_dir / f"per_sample_metrics_{args.split}.csv",
                           ids=ids)
    with open(run_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump({"split": args.split, "threshold": args.threshold,
                   "n": int(len(Y)), **overall}, f, indent=2)

    print(f"Wrote qual_grid_{args.split}.png, "
          f"per_sample_metrics_{args.split}.csv, "
          f"metrics_{args.split}.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (CSIFrameDataset, compute_norm_stats, save_norm_stats)
from metrics import compute_all_metrics
from Stage2.model_stage2   import StaticReconstructor


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir",    type=str, default="/mnt/user-data/uploads")
    p.add_argument("--out_dir",     type=str, default="./runs/stage2_d32")
    # Model
    p.add_argument("--latent_dim",  type=int, default=32, choices=[16, 32, 64])
    p.add_argument("--mask_size",   type=int, default=32)
    p.add_argument("--dropout",     type=float, default=0.2)
    # Optimization
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--patience",    type=int,   default=15,
                   help="Early-stop patience (epochs without val-IoU improvement)")
    p.add_argument("--lr_patience", type=int,   default=5,
                   help="ReduceLROnPlateau patience")
    p.add_argument("--pos_weight",  type=float, default=None,
                   help="Optional BCE pos_weight for class imbalance "
                        "(e.g. ~6.2 for 14%% foreground)")
    # Misc
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--device",      type=str,   default=None,
                   help="cpu / cuda / mps; auto-detect if omitted")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def pick_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float, loss_fn) -> tuple[float, dict]:
    """Return (mean_loss, metrics_dict) for a full pass over `loader`."""
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n_samples  += bsz
        all_y.append(y.cpu())
        all_p.append(torch.sigmoid(logits).cpu())

    mean_loss = total_loss / max(n_samples, 1)
    y_cat = torch.cat(all_y, dim=0)
    p_cat = torch.cat(all_p, dim=0)
    metrics = compute_all_metrics(y_cat, p_cat, threshold=threshold)
    return mean_loss, metrics


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalization stats (train only, saved for reuse) ──
    X_path = data_dir / "X_frame.npy"
    Y_path = data_dir / "Y_frame.npy"
    norm_stats = compute_norm_stats(X_path, data_dir / "frame_train_idx.npy")
    save_norm_stats(norm_stats, out_dir / "norm_stats.json")

    # ── Datasets and loaders ──
    ds_tr = CSIFrameDataset(X_path, Y_path, data_dir / "frame_train_idx.npy", norm_stats)
    ds_va = CSIFrameDataset(X_path, Y_path, data_dir / "frame_val_idx.npy",   norm_stats)
    ds_te = CSIFrameDataset(X_path, Y_path, data_dir / "frame_test_idx.npy",  norm_stats)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # ── Model, loss, optimizer, scheduler ──
    model = StaticReconstructor(
        latent_dim=args.latent_dim, mask_size=args.mask_size, dropout=args.dropout
    ).to(device)

    pos_weight = (torch.tensor([args.pos_weight], device=device)
                  if args.pos_weight is not None else None)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=args.lr_patience
    )

    # ── Save run config for reproducibility ──
    config = {
        **vars(args),
        "device": str(device),
        "num_params": model.num_params(),
        "norm_stats": norm_stats,
        "n_train": len(ds_tr), "n_val": len(ds_va), "n_test": len(ds_te),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {device}")
    print(f"Model params: {model.num_params():,}")
    print(f"Splits: train={len(ds_tr)} | val={len(ds_va)} | test={len(ds_te)}")

    # ── CSV log header ──
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss",
             "val_pixel_acc", "val_iou", "val_dice", "val_mse", "time_s"]
        )

    # ── Training loop ──
    best_iou, best_epoch, epochs_since_improve = -1.0, -1, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        val_loss, val_metrics = evaluate(model, dl_va, device,
                                         args.threshold, loss_fn)
        scheduler.step(val_metrics["iou"])
        epoch_t = time.time() - t0
        current_lr = optim.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, current_lr, f"{train_loss:.5f}", f"{val_loss:.5f}",
                f"{val_metrics['pixel_acc']:.5f}", f"{val_metrics['iou']:.5f}",
                f"{val_metrics['dice']:.5f}", f"{val_metrics['mse']:.5f}",
                f"{epoch_t:.2f}",
            ])

        print(f"[{epoch:3d}/{args.epochs}] "
              f"lr={current_lr:.2e}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={val_metrics['pixel_acc']:.4f}  "
              f"IoU={val_metrics['iou']:.4f}  "
              f"Dice={val_metrics['dice']:.4f}  "
              f"({epoch_t:.1f}s)")

        improved = val_metrics["iou"] > best_iou
        if improved:
            best_iou, best_epoch = val_metrics["iou"], epoch
            epochs_since_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, out_dir / "best.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without "
                      f"val-IoU improvement (best={best_iou:.4f} @ "
                      f"epoch {best_epoch})")
                break

    # ── Save last checkpoint and re-evaluate best on test ──
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "config": config,
    }, out_dir / "last.pt")

    print(f"\nBest val IoU = {best_iou:.4f} at epoch {best_epoch}")

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = evaluate(model, dl_te, device,
                                       args.threshold, loss_fn)
    print(f"Test metrics (best checkpoint): "
          f"loss={test_loss:.4f}  "
          f"acc={test_metrics['pixel_acc']:.4f}  "
          f"IoU={test_metrics['iou']:.4f}  "
          f"Dice={test_metrics['dice']:.4f}  "
          f"MSE={test_metrics['mse']:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"loss": test_loss, **test_metrics,
                   "best_val_iou": best_iou, "best_epoch": best_epoch}, f, indent=2)


if __name__ == "__main__":
    main()

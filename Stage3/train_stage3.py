from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_seq import (CSISeqDataset, compute_norm_stats, save_norm_stats,
                         variant_files)
from metrics import compute_all_metrics
from model_stage3 import build_model, load_stage2_encoder_weights


# ─────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data / variant
    p.add_argument("--variant",     type=str, default="a", choices=["a", "b"])
    p.add_argument("--data_dir",    type=str, default="/mnt/user-data/uploads")
    p.add_argument("--out_dir",     type=str, default="./runs/stage3_a_d32")
    # Model
    p.add_argument("--latent_dim",  type=int, default=32, choices=[16, 32, 64])
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--bidirectional", dest="bidirectional",
                   action="store_true", default=None,
                   help="Force bidirectional. If unset: False for variant a, "
                        "True for variant b (the documented design choices).")
    p.add_argument("--unidirectional", dest="bidirectional",
                   action="store_false",
                   help="Force unidirectional (overrides the variant default).")
    p.add_argument("--mask_size",   type=int,   default=32)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--lstm_dropout",type=float, default=0.0)
    p.add_argument("--prev_embed_dim", type=int, default=64,
                   help="Variant B: embedding width for the previous frame "
                        "Ŷ_{t-1} fed to the autoregressive decoder.")
    p.add_argument("--teacher_forcing_ratio", type=float, default=1.0,
                   help="Variant B: probability of feeding ground-truth "
                        "Y_{t-1} during training (1.0 = pure teacher forcing, "
                        "<1.0 = scheduled sampling). Ignored for Variant A.")
    # Optional warm-start (default: from scratch)
    p.add_argument("--init_encoder_from", type=str, default=None,
                   help="Path to a Stage 2 best.pt to warm-start the encoder.")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder weights (only with --init_encoder_from).")
    # Optimization
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--patience",    type=int,   default=15,
                   help="Early-stop patience (epochs without val-IoU gain).")
    p.add_argument("--lr_patience", type=int,   default=5,
                   help="ReduceLROnPlateau patience.")
    p.add_argument("--pos_weight",  type=float, default=None,
                   help="BCE pos_weight for class imbalance (≈6.24 fully "
                        "balances the ~13.8%% foreground; default: vanilla BCE).")
    # Inference / misc
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--in_memory",   action="store_true", default=True)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args, stats):
    Xf, Yf = variant_files(args.variant)
    Xp, Yp = f"{args.data_dir}/{Xf}", f"{args.data_dir}/{Yf}"
    common = dict(norm_stats=stats, in_memory=args.in_memory)
    tr = CSISeqDataset(Xp, Yp, f"{args.data_dir}/seq_train_idx.npy", **common)
    va = CSISeqDataset(Xp, Yp, f"{args.data_dir}/seq_val_idx.npy",   **common)
    te = CSISeqDataset(Xp, Yp, f"{args.data_dir}/seq_test_idx.npy",  **common)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, drop_last=False)
    return dl(tr, True), dl(va, False), dl(te, False), (len(tr), len(va), len(te))


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              teacher_forcing_ratio: float = 1.0):
    """One pass. Returns (mean_loss, metrics_dict) computed on the full split.

    Metrics are computed on probabilities aggregated over the whole split; for
    Variant B the (B,T,W,W) tensors are flattened to (B*T,W,W) so metrics.py is
    used unchanged and IoU is frame-averaged (IoU_seq, 3.6.3).

    The model receives (x, y, teacher_forcing_ratio). Variant A ignores y/ratio;
    Variant B uses teacher forcing only while model.training is True (so val/
    test passes are pure autoregressive — true inference behaviour).
    """
    model.train(train)
    total_loss, n = 0.0, 0
    probas, targets = [], []

    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, y, teacher_forcing_ratio)   # A:(B,W,W)  B:(B,T,W,W)
        loss = criterion(logits, y)                   # mean reduction → L_seq for B

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        with torch.no_grad():
            p = torch.sigmoid(logits)
            if p.dim() == 4:                    # Variant B → flatten time
                p = p.reshape(-1, p.shape[-2], p.shape[-1])
                yt = y.reshape(-1, y.shape[-2], y.shape[-1])
            else:
                yt = y
            probas.append(p.cpu())
            targets.append(yt.cpu())
    torch.set_grad_enabled(True)

    probas = torch.cat(probas, 0)
    targets = torch.cat(targets, 0)
    metrics = compute_all_metrics(targets, probas)
    return total_loss / max(n, 1), metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve directionality default from the variant
    bidir = args.bidirectional
    if bidir is None:
        bidir = (args.variant == "b")

    # Norm stats on train split only (variant-specific X file, same windows)
    Xf, _ = variant_files(args.variant)
    stats = compute_norm_stats(f"{args.data_dir}/{Xf}",
                               f"{args.data_dir}/seq_train_idx.npy")
    save_norm_stats(stats, out / "norm_stats.json")

    tr_loader, va_loader, te_loader, (n_tr, n_va, n_te) = make_loaders(args, stats)

    model = build_model(
        args.variant,
        latent_dim=args.latent_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=bidir,
        mask_size=args.mask_size,
        dropout=args.dropout,
        lstm_dropout=args.lstm_dropout,
        prev_embed_dim=args.prev_embed_dim,
    ).to(device)

    copied = []
    if args.init_encoder_from:
        copied = load_stage2_encoder_weights(model, args.init_encoder_from,
                                              freeze=args.freeze_encoder)
        print(f"Warm-started encoder: copied {len(copied)} tensors from "
              f"{args.init_encoder_from}"
              + ("  (frozen)" if args.freeze_encoder else ""))

    pos_weight = (torch.tensor([args.pos_weight], device=device)
                  if args.pos_weight else None)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    # Persist config
    config = vars(args).copy()
    config.update(dict(bidirectional=bidir, decoder_in=model.decoder_in,
                       n_train=n_tr, n_val=n_va, n_test=n_te,
                       n_params=sum(p.numel() for p in model.parameters()),
                       device=str(device),
                       encoder_warm_started=bool(args.init_encoder_from),
                       encoder_tensors_copied=len(copied)))
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss",
             "val_pixel_acc", "val_iou", "val_dice", "val_mse", "secs"])

    print(f"\nStage 3 Variant {args.variant.upper()} | "
          f"{'bi' if bidir else 'uni'}directional LSTM | "
          f"d={args.latent_dim} H={args.lstm_hidden} "
          f"decoder_in={model.decoder_in} | device={device}")
    print(f"train {n_tr} | val {n_va} | test {n_te} | "
          f"params {config['n_params']:,}\n")

    best_iou, best_epoch, epochs_no_improve = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, tr_loader, criterion, optimizer, device,
                               True, teacher_forcing_ratio=args.teacher_forcing_ratio)
        va_loss, va_m = run_epoch(model, va_loader, criterion, optimizer, device, False)
        scheduler.step(va_m["iou"])
        secs = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr_now:.2e}", f"{tr_loss:.5f}", f"{va_loss:.5f}",
                 f"{va_m['pixel_acc']:.4f}", f"{va_m['iou']:.4f}",
                 f"{va_m['dice']:.4f}", f"{va_m['mse']:.5f}", f"{secs:.1f}"])

        marker = ""
        if va_m["iou"] > best_iou:
            best_iou, best_epoch, epochs_no_improve = va_m["iou"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "val_iou": best_iou},
                       out / "best.pt")
            marker = "  *best"
        else:
            epochs_no_improve += 1

        print(f"ep {epoch:3d} | lr {lr_now:.1e} | "
              f"train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"IoU {va_m['iou']:.4f} Dice {va_m['dice']:.4f} | "
              f"{secs:.1f}s{marker}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val IoU {best_iou:.4f} @ epoch {best_epoch}).")
            break

    torch.save({"model": model.state_dict(), "config": config,
                "epoch": epoch, "val_iou": va_m["iou"]}, out / "last.pt")

    # Test the best checkpoint
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_m = run_epoch(model, te_loader, criterion, optimizer, device, False)
    te_m["test_loss"] = te_loss
    te_m["best_epoch"] = best_epoch
    te_m["best_val_iou"] = best_iou
    with open(out / "test_metrics.json", "w") as f:
        json.dump(te_m, f, indent=2)

    print(f"\nBest epoch {best_epoch} | val IoU {best_iou:.4f}")
    print(f"TEST  IoU {te_m['iou']:.4f}  Dice {te_m['dice']:.4f}  "
          f"PixAcc {te_m['pixel_acc']:.4f}  MSE {te_m['mse']:.5f}")
    print(f"Artifacts written to {out}/")


if __name__ == "__main__":
    main()

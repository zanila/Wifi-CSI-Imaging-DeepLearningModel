from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# File-name resolution per variant
# ─────────────────────────────────────────────────────────────

def variant_files(variant: str) -> tuple[str, str]:
    """Return (X_filename, Y_filename) for the given variant."""
    v = variant.lower()
    if v == "a":
        return "X_variant_a.npy", "Y_variant_a.npy"
    if v == "b":
        return "X_seq.npy", "Y_seq.npy"
    raise ValueError(f"Unknown variant {variant!r} (expected 'a' or 'b')")


# ─────────────────────────────────────────────────────────────
# Normalization stats  (train split only)
# ─────────────────────────────────────────────────────────────

def compute_norm_stats(X_path: str | Path, train_idx_path: str | Path) -> dict:
    """Return {'mean': float, 'std': float} computed on the train split."""
    X = np.load(X_path, mmap_mode="r")
    tr = np.load(train_idx_path)
    X_tr = np.asarray(X[tr])                          # (N_train, T, 64, 52)
    mean = float(X_tr.mean())
    std  = float(X_tr.std())
    if std < 1e-8:
        std = 1.0
    return {"mean": mean, "std": std}


def save_norm_stats(stats: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class CSISeqDataset(Dataset):
    """
    Sequence-level CSI ↔ mask dataset for Stage 3.

    Returns
    -------
    x : (T, 64, 52) float32 — normalized CSI window
    y : Variant A → (W, W)    float32 — single binary mask (cast for BCE)
        Variant B → (T, W, W) float32 — one binary mask per time step
    """

    def __init__(
        self,
        X_path: str | Path,
        Y_path: str | Path,
        idx_path: str | Path,
        norm_stats: dict,
        in_memory: bool = True,
    ):
        self.idx = np.load(idx_path)

        if in_memory:
            X_all = np.load(X_path)
            Y_all = np.load(Y_path)
            self.X = X_all[self.idx].astype(np.float32, copy=False)
            self.Y = Y_all[self.idx].astype(np.float32, copy=False)
            self.in_memory = True
        else:
            self.X = np.load(X_path, mmap_mode="r")
            self.Y = np.load(Y_path, mmap_mode="r")
            self.in_memory = False

        self.mean = float(norm_stats["mean"])
        self.std  = float(norm_stats["std"])

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        if self.in_memory:
            x = self.X[i]
            y = self.Y[i]
        else:
            j = int(self.idx[i])
            x = np.asarray(self.X[j], dtype=np.float32)
            y = np.asarray(self.Y[j], dtype=np.float32)

        # z-score (same constants applied to every frame of the window)
        x = (x - self.mean) / self.std

        x_t = torch.from_numpy(np.ascontiguousarray(x)).float()   # (T, 64, 52)
        y_t = torch.from_numpy(np.ascontiguousarray(y)).float()   # (W,W) or (T,W,W)
        return x_t, y_t


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
    for variant in ("a", "b"):
        Xf, Yf = variant_files(variant)
        Xp, Yp = f"{base}/{Xf}", f"{base}/{Yf}"
        stats = compute_norm_stats(Xp, f"{base}/seq_train_idx.npy")
        print(f"\n[Variant {variant.upper()}] norm stats: {stats}")

        ds_tr = CSISeqDataset(Xp, Yp, f"{base}/seq_train_idx.npy", stats)
        ds_va = CSISeqDataset(Xp, Yp, f"{base}/seq_val_idx.npy",   stats)
        ds_te = CSISeqDataset(Xp, Yp, f"{base}/seq_test_idx.npy",  stats)
        print(f"  train {len(ds_tr)} | val {len(ds_va)} | test {len(ds_te)}")

        x, y = ds_tr[0]
        print(f"  x: {tuple(x.shape)} {x.dtype} range [{x.min():.3f},{x.max():.3f}] "
              f"mean {x.mean():.3f}")
        print(f"  y: {tuple(y.shape)} {y.dtype} unique {torch.unique(y).tolist()}")

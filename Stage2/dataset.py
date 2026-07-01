from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
# Normalization stats
# ─────────────────────────────────────────────────────────────

def compute_norm_stats(X_path: str | Path, train_idx_path: str | Path) -> dict:
    """Return {'mean': float, 'std': float} computed on the train split."""
    X = np.load(X_path, mmap_mode="r")
    tr = np.load(train_idx_path)
    X_tr = X[tr]                                     # (N_train, T, F)
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

class CSIFrameDataset(Dataset):
    """
    Frame-level CSI ↔ mask dataset for Stage 2.

    Returns
    -------
    x : (T, F) float32 tensor — normalized CSI tensor
    y : (W, W) float32 tensor — binary mask, cast to float for BCE
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
            # Index now, keep only the relevant slice in memory.
            self.X = X_all[self.idx].astype(np.float32, copy=False)
            self.Y = Y_all[self.idx].astype(np.float32, copy=False)
        else:
            # mmap path — leaves arrays on disk, indexed on the fly.
            self.X = np.load(X_path, mmap_mode="r")
            self.Y = np.load(Y_path, mmap_mode="r")
            self._mmap = True

        self.in_memory = in_memory
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

        # z-score
        x = (x - self.mean) / self.std

        x_t = torch.from_numpy(np.ascontiguousarray(x)).float()   # (T, F)
        y_t = torch.from_numpy(np.ascontiguousarray(y)).float()   # (W, W)
        return x_t, y_t


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "dataset_output"
    X_path = f"{base}/X_frame.npy"
    Y_path = f"{base}/Y_frame.npy"

    stats = compute_norm_stats(X_path, f"{base}/frame_train_idx.npy")
    print(f"Train-set norm stats: {stats}")

    ds_tr = CSIFrameDataset(X_path, Y_path, f"{base}/frame_train_idx.npy", stats)
    ds_va = CSIFrameDataset(X_path, Y_path, f"{base}/frame_val_idx.npy",   stats)
    ds_te = CSIFrameDataset(X_path, Y_path, f"{base}/frame_test_idx.npy",  stats)

    print(f"train: {len(ds_tr)}  |  val: {len(ds_va)}  |  test: {len(ds_te)}")
    x, y = ds_tr[0]
    print(f"sample x: shape {tuple(x.shape)}  dtype {x.dtype}  "
          f"range [{x.min():.3f}, {x.max():.3f}]  mean {x.mean():.3f}")
    print(f"sample y: shape {tuple(y.shape)}  dtype {y.dtype}  "
          f"unique {torch.unique(y).tolist()}")

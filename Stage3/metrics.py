from __future__ import annotations

import torch


_EPS = 1e-6


def _binarize(y_pred_proba: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (y_pred_proba >= threshold).to(y_pred_proba.dtype)


def pixel_accuracy(y_true: torch.Tensor, y_pred_proba: torch.Tensor,
                   threshold: float = 0.5) -> torch.Tensor:
    """Acc = (TP + TN) / (TP + TN + FP + FN) averaged over the batch."""
    y_pred = _binarize(y_pred_proba, threshold)
    correct = (y_pred == y_true).float()              # (B, W, W)
    return correct.mean(dim=(-2, -1)).mean()


def iou_score(y_true: torch.Tensor, y_pred_proba: torch.Tensor,
              threshold: float = 0.5, eps: float = _EPS) -> torch.Tensor:
    """IoU = |A ∩ B| / |A ∪ B| averaged over the batch."""
    y_pred = _binarize(y_pred_proba, threshold)
    inter = (y_pred * y_true).sum(dim=(-2, -1))
    union = y_pred.sum(dim=(-2, -1)) + y_true.sum(dim=(-2, -1)) - inter
    iou = (inter + eps) / (union + eps)               # (B,)
    return iou.mean()


def dice_coefficient(y_true: torch.Tensor, y_pred_proba: torch.Tensor,
                     threshold: float = 0.5, eps: float = _EPS) -> torch.Tensor:
    """Dice = 2 |A ∩ B| / (|A| + |B|) averaged over the batch."""
    y_pred = _binarize(y_pred_proba, threshold)
    inter = (y_pred * y_true).sum(dim=(-2, -1))
    size  = y_pred.sum(dim=(-2, -1)) + y_true.sum(dim=(-2, -1))
    dice = (2 * inter + eps) / (size + eps)
    return dice.mean()


def mse(y_true: torch.Tensor, y_pred_proba: torch.Tensor) -> torch.Tensor:
    """Pixel-wise MSE on raw probabilities (not thresholded)."""
    return ((y_true - y_pred_proba) ** 2).mean()


# ─────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: torch.Tensor, y_pred_proba: torch.Tensor,
                        threshold: float = 0.5) -> dict[str, float]:
    """Return a dict of all metrics. Inputs are full-dataset tensors."""
    return {
        "pixel_acc": float(pixel_accuracy(y_true, y_pred_proba, threshold)),
        "iou":       float(iou_score(y_true, y_pred_proba, threshold)),
        "dice":      float(dice_coefficient(y_true, y_pred_proba, threshold)),
        "mse":       float(mse(y_true, y_pred_proba)),
    }

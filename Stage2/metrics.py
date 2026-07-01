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
        "pixel_acc": pixel_accuracy(y_true, y_pred_proba, threshold).item(),
        "iou":       iou_score(y_true, y_pred_proba, threshold).item(),
        "dice":      dice_coefficient(y_true, y_pred_proba, threshold).item(),
        "mse":       mse(y_true, y_pred_proba).item(),
    }


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # Perfect prediction → all metrics should be ~1.0 (except MSE ~ 0)
    y_true = torch.randint(0, 2, (8, 32, 32)).float()
    y_pred_perfect = y_true.clone()
    print("Perfect prediction:", compute_all_metrics(y_true, y_pred_perfect))

    # Constant 0.5 baseline — IoU/Dice/Acc all degrade
    y_pred_half = torch.full_like(y_true, 0.5)
    print("Constant 0.5    :", compute_all_metrics(y_true, y_pred_half))

    # Inverted prediction → IoU ≈ 0, Dice ≈ 0, Acc ≈ 0
    y_pred_inv = 1.0 - y_true
    print("Inverted        :", compute_all_metrics(y_true, y_pred_inv))

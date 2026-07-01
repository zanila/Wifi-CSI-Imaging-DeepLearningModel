from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────

class CSIEncoder(nn.Module):
    """
    2D-CNN encoder that treats the CSI tensor (T × F) as a single-channel
    image. Three stride-2 conv blocks progressively reduce spatial size and
    increase channel depth, followed by global average pooling and an FC
    projection to latent dim d.

    Input  : (B, 1, T=64, F=52)
    Output : (B, d)        with d ∈ {16, 32, 64}

    Spatial trace (T=64, F=52):
      conv1 stride 2 →  (32, 26)   channels  1 → 32
      conv2 stride 2 →  (16, 13)   channels 32 → 64
      conv3 stride 2 →  ( 8,  7)   channels 64 → 128
      global avg pool → (128,)
      fc              → (d,)
    """

    def __init__(self, latent_dim: int = 32, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.conv1 = block(in_channels, 32)
        self.conv2 = block(32, 64)
        self.conv3 = block(64, 128)
        self.gap   = nn.AdaptiveAvgPool2d(1)        # (B, 128, 1, 1)
        self.fc    = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → add channel dim if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)                       # (B, 1, T, F)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).flatten(1)                   # (B, 128)
        z = self.fc(x)                               # (B, d)
        return z


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class MaskDecoder(nn.Module):
    """
    MLP decoder mapping latent z ∈ R^d to a W×W mask of LOGITS
    (sigmoid applied externally for numerical stability).

    Architecture: d → 256 → 512 → W*W
    Dropout in the wider hidden layers to mitigate overfitting
    on the small (~850-sample) training set.
    """

    def __init__(self, latent_dim: int = 32, mask_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.mask_size = mask_size
        out_dim = mask_size * mask_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, d) → (B, W, W) logits
        logits_flat = self.net(z)
        logits = logits_flat.view(-1, self.mask_size, self.mask_size)
        return logits


# ─────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────

class StaticReconstructor(nn.Module):
    """
    Stage 2 model: CSI tensor → silhouette mask.

      forward(x)        → logits  (use with BCEWithLogitsLoss)
      predict_proba(x)  → probabilities ∈ [0,1]^(W×W)
      predict(x, τ)     → binary mask ∈ {0,1}^(W×W) at threshold τ
    """

    def __init__(self, latent_dim: int = 32, mask_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.encoder = CSIEncoder(latent_dim=latent_dim)
        self.decoder = MaskDecoder(latent_dim=latent_dim, mask_size=mask_size,
                                   dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).to(torch.uint8)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = StaticReconstructor(latent_dim=32, mask_size=32)
    print(model)
    print(f"\nTrainable parameters: {model.num_params():,}")

    x = torch.randn(4, 64, 52)              # batch of 4 CSI tensors
    logits = model(x)
    proba  = model.predict_proba(x)
    mask   = model.predict(x)
    print(f"\ninput  : {tuple(x.shape)}")
    print(f"logits : {tuple(logits.shape)}   range [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"proba  : {tuple(proba.shape)}   range [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"mask   : {tuple(mask.shape)}   dtype {mask.dtype}")

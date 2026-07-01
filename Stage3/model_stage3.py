from __future__ import annotations

import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════
# REUSED UNCHANGED FROM STAGE 2  (model.py)
# ═════════════════════════════════════════════════════════════

class CSIEncoder(nn.Module):
    """
    2D-CNN encoder that treats the CSI tensor (T × F) as a single-channel
    image. Three stride-2 conv blocks progressively reduce spatial size and
    increase channel depth, followed by global average pooling and an FC
    projection to latent dim d.

    Input  : (B, 1, T=64, F=52)   (channel dim added on the fly if missing)
    Output : (B, d)               with d ∈ {16, 32, 64}

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


class MaskDecoder(nn.Module):
    """
    MLP decoder mapping a latent vector ∈ R^(in_dim) to a W×W mask of LOGITS
    (sigmoid applied externally for numerical stability).

    Architecture: in_dim → 256 → 512 → W*W
    Dropout in the wider hidden layers to mitigate overfitting on the small
    training set. In Stage 2 `in_dim == latent_dim`; in Stage 3 `in_dim` is
    the LSTM output width (H or 2H). The hidden widths (256, 512) and dropout
    are unchanged from Stage 2.
    """

    def __init__(self, latent_dim: int = 32, mask_size: int = 32,
                 dropout: float = 0.2):
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
        logits_flat = self.net(z)                                   # (N, W*W)
        logits = logits_flat.view(-1, self.mask_size, self.mask_size)
        return logits                                               # (N, W, W)


# ═════════════════════════════════════════════════════════════
# NEW IN STAGE 3 — separate decoder for Variant B
# ═════════════════════════════════════════════════════════════

class AutoregressiveMaskDecoder(nn.Module):
    """
    Variant B decoder implementing  Ŷ_t = σ(D_ϕ^AR(h_t, Ŷ_{t-1})).

    Distinct from the Stage 2 MaskDecoder: it conditions each frame on the
    previous frame. The previous mask Ŷ_{t-1} ∈ [0,1]^(W×W) is flattened and
    embedded, then concatenated with the LSTM context h_t before the same MLP
    body used in Stage 2 (256 → 512 → W*W). Outputs LOGITS.

      h_t           : (N, context_dim)   LSTM output at step t (2H if bi)
      prev_flat     : (N, W*W)           previous mask, {0,1} (teacher) or
                                          binarized own prediction (inference)
    """

    def __init__(self, context_dim: int, mask_size: int = 32,
                 prev_embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mask_size = mask_size
        out_dim = mask_size * mask_size

        self.prev_proj = nn.Sequential(
            nn.Linear(out_dim, prev_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Linear(context_dim + prev_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, h_t: torch.Tensor, prev_flat: torch.Tensor) -> torch.Tensor:
        z = torch.cat([h_t, self.prev_proj(prev_flat)], dim=1)
        logits_flat = self.net(z)
        return logits_flat.view(-1, self.mask_size, self.mask_size)


# ═════════════════════════════════════════════════════════════
# NEW IN STAGE 3
# ═════════════════════════════════════════════════════════════

class _TemporalBase(nn.Module):
    """Shared encoder + LSTM machinery. Each variant builds its own decoder."""

    def __init__(self,
                 latent_dim: int = 32,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 1,
                 bidirectional: bool = False,
                 mask_size: int = 32,
                 dropout: float = 0.2,
                 lstm_dropout: float = 0.0):
        super().__init__()
        self.latent_dim    = latent_dim
        self.lstm_hidden   = lstm_hidden
        self.lstm_layers   = lstm_layers
        self.bidirectional = bidirectional
        self.mask_size     = mask_size
        self.dropout       = dropout

        # E_θ — reused, shared across all T frames
        self.encoder = CSIEncoder(latent_dim=latent_dim)

        # LSTM — the genuinely new sequence component
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Width of the per-step LSTM context fed to the decoder
        self.decoder_in = lstm_hidden * (2 if bidirectional else 1)

    # ---- helpers -----------------------------------------------------
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, F_T, F_S)  →  embeddings (B, T, d)

        The CNN is applied independently to each of the T frames by folding
        the time axis into the batch axis, then unfolding.
        """
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B * T, *x.shape[2:])           # (B*T, 64, 52)
        e = self.encoder(x)                          # (B*T, d)
        return e.view(B, T, self.latent_dim)         # (B, T, d)

    def predict_proba(self, x: torch.Tensor, *args, **kw) -> torch.Tensor:
        """Apply sigmoid to the forward logits (inference only)."""
        return torch.sigmoid(self.forward(x, *args, **kw))


class Stage3VariantA(_TemporalBase):
    """
    Sequence-to-one (3.5, Variant A) — the controlled "isolate the LSTM" model.

    Unidirectional LSTM; the final hidden state h_T of the top layer is the
    temporally-enriched latent that the REUSED Stage 2 MaskDecoder maps to a
    single mask for the last frame of the window (target_idx_in_window = 9).

    forward(x, *_unused) : (B, T, 64, 52) → logits (B, W, W)
    """

    def __init__(self, prev_embed_dim: int = 64, **kw):
        # prev_embed_dim accepted for a uniform build_model signature; unused.
        kw.setdefault("bidirectional", False)        # unidirectional per 3.3.3
        super().__init__(**kw)
        self.decoder = MaskDecoder(latent_dim=self.decoder_in,
                                   mask_size=self.mask_size, dropout=self.dropout)

    def forward(self, x: torch.Tensor, y=None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        e = self.encode_sequence(x)                  # (B, T, d)
        out, (h_n, _) = self.lstm(e)                 # h_n: (L*dir, B, H)
        h_last = h_n[-1]                             # top layer, last step (B, H)
        logits = self.decoder(h_last)                # (B, W, W)
        return logits


class Stage3VariantB(_TemporalBase):
    """
    Sequence-to-sequence (3.5, Variant B) — faithful autoregressive decoder.

    Bidirectional LSTM (design choice: better IoU) provides a per-step context
    h_t that has seen the whole CSI window. A SEPARATE autoregressive decoder
    produces one mask per step, conditioning each on the previous frame:

        Ŷ_t = σ(D_ϕ^AR(h_t, Ŷ_{t-1})).

    Training uses teacher forcing — the ground-truth Y_{t-1} is fed in with
    probability `teacher_forcing_ratio` (default 1.0); otherwise the model's
    own (binarized, detached) prediction is used (scheduled sampling). At
    inference (model.eval()), the model always uses its own prediction. The
    first step is seeded with a zero "start" frame.

    forward(x, y, teacher_forcing_ratio) : (B, T, 64, 52) → logits (B, T, W, W)
      y : (B, T, W, W) ground-truth masks; required for teacher forcing,
          ignored when the model is in eval mode or y is None.
    """

    def __init__(self, prev_embed_dim: int = 64, **kw):
        kw.setdefault("bidirectional", True)         # bidirectional per design
        super().__init__(**kw)
        self.prev_embed_dim = prev_embed_dim
        self.decoder = AutoregressiveMaskDecoder(
            context_dim=self.decoder_in, mask_size=self.mask_size,
            prev_embed_dim=prev_embed_dim, dropout=self.dropout)

    def forward(self, x: torch.Tensor, y=None,
                teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        W = self.mask_size
        e = self.encode_sequence(x)                  # (B, T, d)
        out, _ = self.lstm(e)                        # (B, T, dir*H)

        prev = x.new_zeros(B, W * W)                 # start frame = zeros
        use_tf = self.training and (y is not None)
        logits_seq = []
        for t in range(T):
            logits_t = self.decoder(out[:, t], prev)         # (B, W, W)
            logits_seq.append(logits_t)
            if t < T - 1:
                if use_tf and (torch.rand(1).item() < teacher_forcing_ratio):
                    prev = y[:, t].reshape(B, W * W)         # ground-truth {0,1}
                else:
                    prev = (torch.sigmoid(logits_t) >= 0.5).float() \
                              .reshape(B, W * W).detach()     # own prediction
        return torch.stack(logits_seq, dim=1)        # (B, T, W, W)


# ═════════════════════════════════════════════════════════════
# Factory + optional warm-start
# ═════════════════════════════════════════════════════════════

def build_model(variant: str, **kw) -> _TemporalBase:
    """variant ∈ {'a', 'b'}. Extra kwargs forwarded to the variant class."""
    variant = variant.lower()
    if variant == "a":
        return Stage3VariantA(**kw)
    if variant == "b":
        return Stage3VariantB(**kw)
    raise ValueError(f"Unknown variant {variant!r} (expected 'a' or 'b')")


def load_stage2_encoder_weights(model: _TemporalBase,
                                stage2_ckpt_path: str,
                                freeze: bool = False) -> list[str]:
    """
    Optional warm-start: copy the CNN encoder weights from a Stage 2 best.pt
    into this model's encoder. The design default is FROM SCRATCH; this helper
    exists only for the ablation. Returns the list of successfully copied keys.

    Accepts a checkpoint that is either a raw state_dict or a dict that wraps
    the weights under a common key ('model_state', 'model', 'state_dict',
    'model_state_dict', 'net', 'weights'). Encoder keys are matched by the
    'encoder.' prefix.
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    sd = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state", "model", "state_dict",
                    "model_state_dict", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break

    enc_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            enc_sd[k[len("encoder."):]] = v
        elif k.startswith("enc."):
            enc_sd[k[len("enc."):]] = v
    if not enc_sd:
        # Stage 2 StaticReconstructor may store encoder under a different name;
        # fall back to matching by trailing parameter names.
        own = model.encoder.state_dict()
        enc_sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}

    missing = model.encoder.load_state_dict(enc_sd, strict=False)
    copied = [k for k in model.encoder.state_dict() if k in enc_sd]

    if freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False

    return copied


# ═════════════════════════════════════════════════════════════
# Smoke test
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, T, Tf, Fs, W = 4, 10, 64, 52, 32
    x = torch.randn(B, T, Tf, Fs)
    y_seq = (torch.rand(B, T, W, W) > 0.8).float()

    a = build_model("a", latent_dim=32, lstm_hidden=64)
    ya = a(x)
    pa = a.predict_proba(x)
    print(f"Variant A  logits {tuple(ya.shape)}  proba range "
          f"[{pa.min():.3f}, {pa.max():.3f}]  "
          f"params {sum(p.numel() for p in a.parameters()):,}")
    assert ya.shape == (B, W, W)

    b = build_model("b", latent_dim=32, lstm_hidden=64)   # bidirectional + AR
    b.train()
    yb_tf = b(x, y_seq, teacher_forcing_ratio=1.0)        # teacher forcing path
    b.eval()
    yb = b(x)                                             # pure autoregressive
    pb = torch.sigmoid(yb)
    print(f"Variant B  logits {tuple(yb.shape)}  proba range "
          f"[{pb.min():.3f}, {pb.max():.3f}]  "
          f"params {sum(p.numel() for p in b.parameters()):,}")
    assert yb_tf.shape == (B, T, W, W) and yb.shape == (B, T, W, W)

    # Strict controlled-comparison config: decoder dimensionally == Stage 2
    a32 = build_model("a", latent_dim=32, lstm_hidden=32)
    assert a32.decoder_in == 32
    print("Variant A with lstm_hidden=latent_dim → decoder_in =", a32.decoder_in,
          "(matches Stage 2 decoder)")
    print("OK")

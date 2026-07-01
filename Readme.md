# Stage 1 – Section 3.3.1: CSI Tensor Construction
Implements the CSI tensor construction pipeline as described in the
methodology. The continuous CSI stream is partitioned into fixed-duration
temporal windows (default 1 second). Each window is divided into T
equally-spaced temporal bins, and raw packets falling into each bin are
aggregated to produce a fixed-shape amplitude tensor:

&nbsp;&nbsp;&nbsp;&nbsp;`x ∈ ℝ^(T×S) (single-antenna setup, P = 1 collapsed)`

Two aggregation strategies are supported for bins containing multiple packets:

&nbsp;&nbsp;&nbsp;&nbsp;\- mean: Average amplitude across all packets in the bin.  
&nbsp;&nbsp;&nbsp;&nbsp;\- max: Maximum (max-pooled) amplitude across all packets in the bin.

Three imputation strategies are supported for empty bins:

&nbsp;&nbsp;&nbsp;&nbsp;\- zero: Fill the bin with zeros.  
&nbsp;&nbsp;&nbsp;&nbsp;\- linear: Linearly interpolate values from neighboring non-empty bins.  
&nbsp;&nbsp;&nbsp;&nbsp;\- nearest: Copy values from the nearest non-empty bin. 

The module also handles CSI-to-image pairing: for each temporal window
the image whose timestamp is closest to the window centre is selected.

### Usage:
    python csi_tensor_construction.py \
        --csi_csv    path/to/csi.csv \
        --csi_npy    path/to/csiComplex.npy \
        --image_dir  path/to/images/ \
        --output_dir path/to/output/ \
        --window_sec 1.0 \
        --stride_sec 0.5 \
        --T 64 \
        --aggregation mean \
        --imputation  linear


# Stage 1 – Section 3.3.2: Ground-Truth Extraction from Video
Converts raw RGB video frames into binary silhouette maps `Y_t ∈ {0,1}^{W×W}`
using YOLOv8 instance segmentation.

### Pipeline per frame:
1. YOLOv8-seg detects person → pixel-level segmentation mask
2. Bounding box of detected person is extracted
3. Bounding box is expanded to a square (with configurable padding
4. Mask is cropped to the square bounding box\
5. Cropped mask is resized to W×W (default 32×32)
6. Thresholded to binary {0, 1}

### Outputs:
&nbsp;&nbsp;&nbsp;&nbsp;\- Y_masks.npy           : (N_images, W, W) uint8 array of binary masks  
&nbsp;&nbsp;&nbsp;&nbsp;\- image_ids.npy         : (N_images,) array of image IDs (int) matching the masks  
&nbsp;&nbsp;&nbsp;&nbsp;\- mask_metadata.json    : per-image metadata (bbox, confidence, fg_ratio, etc.)  
&nbsp;&nbsp;&nbsp;&nbsp;\- qa_grid.png           : visual QA grid showing original → full mask → 32×32 mask  

### Requirements:
    pip install ultralytics opencv-python numpy Pillow

### Usage:
    python ground_truth_extraction.py \
        --image_dir /path/to/wificam/j3/640/ \
        --output_dir /path/to/output/ \
        --mask_size 32 \
        --bbox_padding 0.1 \
        --confidence_threshold 0.5 \
        --yolo_model yolov8n-seg.pt


# Stage 1 – Section 3.3.3: Temporal Alignment and Dataset Construction
Builds three supervised datasets from the outputs of 3.3.1 and 3.3.2:

&nbsp;&nbsp;&nbsp;&nbsp;\- Stage 2 (static):           D_frame      — 1 CSI tensor → 1 mask  
&nbsp;&nbsp;&nbsp;&nbsp;\- Stage 3 Variant A (many→1): D_variant_a  — T CSI tensors → 1 mask (last)  
&nbsp;&nbsp;&nbsp;&nbsp;\- Stage 3 Variant B (many→N): D_seq        — T CSI tensors → T masks  

### Inputs (from prior stages):
&nbsp;&nbsp;&nbsp;&nbsp;\- X_csi_tensors.npy : (N_csi, T_csi, F)     CSI tensors from 3.3.1  
&nbsp;&nbsp;&nbsp;&nbsp;\- y_image_ids.npy   : (N_csi,)              CSI→image mapping from 3.3.1  
&nbsp;&nbsp;&nbsp;&nbsp;\- Y_masks.npy       : (N_img, W, W)         binary masks from 3.3.2  
&nbsp;&nbsp;&nbsp;&nbsp;\- image_ids.npy     : (N_img,)              image IDs for the masks  

### Usage:
      python build_datasets_333.py \
          --csi_tensors    ./X_csi_tensors.npy \
          --csi_image_ids  ./y_image_ids.npy \
          --masks          ./Y_masks.npy \
          --mask_image_ids ./image_ids.npy \
          --output_dir     ./datasets/ \
          --seq_len 10 \
          --stride 1 \
          --write_splits


# Stage 2 — Static Encoder–Decoder Reconstruction
Maps a single CSI tensor `X_t ∈ R^(T×F)` to a binary silhouette
`Ŷ_t ∈ {0,1}^(W×W)` without temporal modelling. Baseline for the
Stage 3 temporal model.

## Files

| File | Purpose |
|---|---|
| `model_stage2.py` | `StaticReconstructor` = `CSIEncoder` (CNN) + `MaskDecoder` (MLP) |
| `dataset.py` | `CSIFrameDataset` + train-only z-score normalization helpers |
| `metrics.py` | Pixel accuracy, IoU, Dice, MSE (per 3.6.2) |
| `train_stage2.py` | Training loop with BCE, Adam, early stopping, CSV log |
| `evaluate_stage2.py` | Test metrics + qualitative grid + per-sample CSV |

## Inputs (from Stage1 - Section 3.3.3)

Expected layout in `--data_dir`:

&nbsp;&nbsp;&nbsp;&nbsp;\- X_frame.npy            (N, T=64, F=52)  float32  
&nbsp;&nbsp;&nbsp;&nbsp;\- Y_frame.npy            (N, W=32, W=32)  uint8 {0,1}  
&nbsp;&nbsp;&nbsp;&nbsp;\- frame_train_idx.npy    (850,)           int64  
&nbsp;&nbsp;&nbsp;&nbsp;\- frame_val_idx.npy      (181,)           int64  
&nbsp;&nbsp;&nbsp;&nbsp;\- frame_test_idx.npy     (183,)           int64  
&nbsp;&nbsp;&nbsp;&nbsp;\- frame_image_ids.npy    (N,)             int64          


## Quick start

```bash
# 1) Train (writes checkpoints + log to ./runs/stage2_d32/)
python train_stage2.py \
    --data_dir /path/to/3.3.3/outputs \
    --out_dir  ./runs/stage2_d32 \
    --latent_dim 32 \
    --epochs 100 --batch_size 32 --lr 1e-3

# 2) Evaluate best checkpoint on test split
python evaluate_stage2.py \
    --data_dir /path/to/3.3.3/outputs \
    --run_dir  ./runs/stage2_d32 \
    --split    test \
    --n_examples 12
```

## Configuration

**Architecture**
- `--latent_dim {16, 32, 64}` — latent dim `d` of the encoder (ablation knob per 3.4.1)
- `--dropout` — MLP decoder dropout (default 0.2)

**Optimization**
- `--lr`, `--weight_decay`, `--batch_size`, `--epochs`
- `--patience` — early-stop patience on val IoU (default 15)
- `--lr_patience` — ReduceLROnPlateau patience (default 5)
- `--pos_weight` — BCE positive-class weight for the ~14% foreground
  imbalance. Set to ~6.2 to fully balance, or leave unset (default) for
  vanilla BCE.

**Inference**
- `--threshold` — sigmoid threshold for binarization (default 0.5)

## Outputs of a training run

```
runs/stage2_d32/
├── config.json              # all hyperparameters + N_train/val/test
├── norm_stats.json          # train-set μ, σ (reused at evaluation)
├── train_log.csv            # per-epoch loss + val metrics
├── best.pt                  # checkpoint with highest val IoU
├── last.pt                  # final-epoch checkpoint
└── test_metrics.json        # test-set metrics of best checkpoint
```

After `evaluate_stage2.py`:

```
├── metrics_test.json
├── per_sample_metrics_test.csv
└── qual_grid_test.png       # CSI | GT | prediction triplets
```

## Notes on design choices

**Normalization.** CSI amplitudes have mean ~11.7 and std ~6.3 with
range [0, 42]. Z-score normalization stats are computed on the **train
split only** and persisted to `norm_stats.json` so that evaluation uses
identical constants (no leakage).

**Loss.** BCE is the methodology default. Implemented as
`BCEWithLogitsLoss` (sigmoid is fused into the loss for numerical
stability); the model returns logits, and inference applies sigmoid
explicitly via `predict_proba()`. Mathematically identical to
`σ(D_ϕ(z))` from 3.4.2.

**Splits.** The 3.3.3 splits are temporal-block-based with a 1-tensor
gap between train/val and val/test to prevent leakage from overlapping
sequences. Stage 2 only needs frame-level splits; the sequence-level
splits are untouched and reserved for Stage 3.

## dataset.py

PyTorch Dataset wrapper for D_frame (Stage 2 input).

&nbsp;&nbsp;&nbsp;&nbsp;X_frame.npy : (N, T, F) float32   — CSI amplitude tensors  
&nbsp;&nbsp;&nbsp;&nbsp;Y_frame.npy : (N, W, W) uint8     — binary silhouette masks  
&nbsp;&nbsp;&nbsp;&nbsp;frame_{train,val,test}_idx.npy    — temporal block-based split indices  

### Normalization
CSI amplitudes have range ~[0, 42] with mean ~11.7 — far from the
typical [0, 1] or zero-mean scale that CNNs train well on. We apply
z-score normalization:

&nbsp;&nbsp;&nbsp;&nbsp;`X_norm = (X - μ_train) / σ_train`

Statistics μ and σ are computed on the train split only to prevent
val/test leakage. They are saved to `norm_stats.json` so the
evaluation script reuses the same constants.

Two scalar stats (global mean, global std) are sufficient here:
the channel dimension is the singleton conv input, and (T, F) is
treated as a 2D spatial grid where per-pixel stats would be both
expensive and inappropriate.

## evaluate_stage2.py

Loads the best checkpoint from a training run, computes test-set metrics (pixel accuracy, IoU, Dice, MSE), and saves qualitative figures:

&nbsp;&nbsp;&nbsp;&nbsp;\- qual_grid.png   : N triplets of (CSI input, GT mask, predicted mask)  
&nbsp;&nbsp;&nbsp;&nbsp;\- test_metrics.json (overwrites/refreshes the one written at train end)  
&nbsp;&nbsp;&nbsp;&nbsp;\- per_sample_metrics.csv  

### Usage:
    python evaluate_stage2.py \
      --data_dir /mnt/user-data/uploads \
      --run_dir  ./runs/stage2_d32 \
      --n_examples 12 \
      --threshold 0.5

## metrics.py
**Section 3.6.2 — Metrics for Static Reconstruction**
Pixel-wise and shape-overlap metrics for binary silhouette maps.

All metrics accept (B, W, W) tensors and return per-batch means.
Binary metrics (acc, IoU, Dice) require thresholding; MSE uses raw probas.

### Notation
&nbsp;&nbsp;&nbsp;&nbsp;\- y_true       : (B, W, W) {0,1} ground-truth mask  
&nbsp;&nbsp;&nbsp;&nbsp;\- y_pred_proba : (B, W, W) [0,1] predicted probability map  
&nbsp;&nbsp;&nbsp;&nbsp;\- y_pred_bin   : (B, W, W) {0,1} thresholded prediction at τ (default 0.5)  


## model_stage2.py
**Section 3.4 — Stage 2: Static Encoder–Decoder Reconstruction**
Maps a single CSI tensor `X_t ∈ R^(T×F)` to a binary silhouette
map `Ŷ_t ∈ [0,1]^(W×W)` without temporal modelling.

&nbsp;&nbsp;&nbsp;&nbsp;\- Encoder:  `z = E_θ(X_t),    z ∈ R^d`        (CNN)  
&nbsp;&nbsp;&nbsp;&nbsp;\- Decoder:  `Ŷ_t = σ(D_ϕ(z)) ∈ [0,1]^(W×W)`    (MLP + sigmoid)  
&nbsp;&nbsp;&nbsp;&nbsp;\- Full:     `Ŷ_t = σ(D_ϕ(E_θ(X_t)))`  

## train_stage2.py
**Section 3.4.3 — Training the Stage 2 Static Encoder–Decoder**

Training loop:

&nbsp;&nbsp;&nbsp;&nbsp;\- Loss:        BCEWithLogitsLoss (numerically stable BCE)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbs;optional pos_weight to counter class imbalance  
&nbsp;&nbsp;&nbsp;&nbsp;\- Optimizer:   Adam with weight decay  
&nbsp;&nbsp;&nbsp;&nbsp;\- Scheduler:   ReduceLROnPlateau on validation IoU  
&nbsp;&nbsp;&nbsp;&nbsp;\- Early stop:  patience epochs without val-IoU improvement  
&nbsp;&nbsp;&nbsp;&nbsp;\- Checkpoint:  best (highest val IoU) and last  
&nbsp;&nbsp;&nbsp;&nbsp;\- Log:         per-epoch CSV with train/val loss + all val metrics  

### Usage:
    python train_stage2.py \
      --data_dir /mnt/user-data/uploads \
      --out_dir  ./runs/stage2_d32 \
      --latent_dim 32 \
      --epochs 100 --batch_size 32 --lr 1e-3


# Stage 3 — Temporal Encoder–LSTM–Decoder Reconstruction
Extends the Stage 2 static model with an LSTM over per-frame CSI embeddings.
The CNN encoder and MLP decoder are **reused from Stage 2**; the LSTM is the
only new component, so any IoU gain over Stage 2 (≈0.549) is attributable to
temporal context.

&nbsp;&nbsp;&nbsp;&nbsp;per frame :  `e_t = E_θ(H_t),  e_t ∈ R^d`                  (CNN, reused)  
&nbsp;&nbsp;&nbsp;&nbsp;sequence  :  `h_t = LSTM(e_t, h_{t-1})`                    (NEW)  
&nbsp;&nbsp;&nbsp;&nbsp;Variant A :  `Ŷ_T = σ(D_ϕ(h_T))`                           (seq → 1 mask, reused decoder)  
&nbsp;&nbsp;&nbsp;&nbsp;Variant B :  `Ŷ_t = σ(D_ϕ^AR(h_t, Ŷ_{t-1})),  t=1..T`       (seq → seq, separate AR decoder)  


## Two different jobs, two different decoder policies

- **Variant A** is the controlled "isolate the LSTM" experiment (Experiment 2,
  3.6.4). It reuses the Stage 2 `MaskDecoder` **unchanged**, so the LSTM is the
  only architectural addition and any IoU gain over Stage 2 is attributable to
  temporal context.
- **Variant B** implements the 3.5 sequence-to-sequence model, with
  a **separate autoregressive decoder** `D_ϕ^AR(h_t, Ŷ_{t-1})` trained with
  teacher forcing. It is conditioning each frame on the previous one requires
  augmenting the decoder input, and that output-space feedback is what produces
  temporal coherence (the quantity TC reports). A bidirectional encoder feeding
  an autoregressive output decoder is the standard seq2seq arrangement.

## Files

| File | Purpose |
|---|---|
| `model_stage3.py` | `CSIEncoder` + `MaskDecoder` (reused) + `AutoregressiveMaskDecoder` (Variant B) + LSTM + `Stage3VariantA/B` + `build_model` |
| `dataset_seq.py` | `CSISeqDataset` (Variant A/B) + train-only z-score normalization |
| `metrics.py` | Pixel acc, IoU, Dice, MSE — **byte-identical to Stage 2** |
| `train_stage3.py` | BCE + Adam + ReduceLROnPlateau, early stop on val IoU, CSV log, checkpoints |
| `evaluate_stage3.py` | Test metrics (+ Temporal Consistency for B), qualitative grid, per-sample CSV |

## Inputs (from Stage1 - Section 3.3.3)

&nbsp;&nbsp;&nbsp;&nbsp;\- X_variant_a.npy   (1207, 10, 64, 52) float32   # Variant A windows  
&nbsp;&nbsp;&nbsp;&nbsp;\- Y_variant_a.npy   (1207, 32, 32)     uint8      # last-frame mask (== Y_seq[:,9])  
&nbsp;&nbsp;&nbsp;&nbsp;\- X_seq.npy         (1207, 10, 64, 52) float32   # Variant B windows (== X_variant_a)  
&nbsp;&nbsp;&nbsp;&nbsp;\- Y_seq.npy         (1207, 10, 32, 32) uint8      # per-step masks  
&nbsp;&nbsp;&nbsp;&nbsp;\- seq_train_idx.npy (834,)  range [0,833]  
&nbsp;&nbsp;&nbsp;&nbsp;\- seq_val_idx.npy   (171,)  range [844,1014]  
&nbsp;&nbsp;&nbsp;&nbsp;\- seq_test_idx.npy  (182,)  range [1025,1206]  

Splits are **contiguous temporal blocks** separated by a 10-sample (= seq_len)
gap with zero index overlap, so the stride-1 window overlap (each window shares
9/10 CSI tensors with its neighbour) does **not** leak across splits. Foreground
ratio ≈ 0.138 → `--pos_weight 6.24` fully balances BCE.

## Design choices

1. **Variant first:** A (seq → 1 mask), then B.
2. **Encoder init:** from scratch (default). `--init_encoder_from <stage2 best.pt>`
   warm-starts the encoder for the optional ablation.
3. **Directionality:** Variant A unidirectional (consistent with last-frame
   supervision, `target_idx_in_window=9`); Variant B bidirectional.

The directionality default is inferred from `--variant` (A→uni, B→bi); override
with `--bidirectional` / `--unidirectional`.

## Quick start

```bash
# Variant A — unidirectional, seq → 1 mask
python train_stage3.py --variant a \
    --data_dir /path/to/uploads --out_dir ./runs/stage3_a_d32 \
    --latent_dim 32 --lstm_hidden 64 --epochs 100 --batch_size 32 --lr 1e-3

python evaluate_stage3.py --data_dir /path/to/uploads \
    --run_dir ./runs/stage3_a_d32 --split test --n_examples 12

# Variant B — bidirectional, seq → seq
python train_stage3.py --variant b \
    --data_dir /path/to/uploads --out_dir ./runs/stage3_b_d32 \
    --latent_dim 32 --lstm_hidden 64

python evaluate_stage3.py --data_dir /path/to/uploads \
    --run_dir ./runs/stage3_b_d32 --split test
```

## Controlled comparison vs Stage 2 (Experiment 2, Section 3.6.4)

For the cleanest "only the LSTM is new" claim, run Variant A with
`--lstm_hidden 32` (= `--latent_dim`). Then the decoder consumes a 32-dim
vector exactly as in Stage 2, so the decoder is dimensionally identical and
the LSTM is the sole architectural addition. With the default `--lstm_hidden
64`, the decoder's *input width* is the only thing that differs from Stage 2
(its hidden widths 256→512→1024 and dropout are unchanged); document whichever
you use.

## Key configurations

- `--latent_dim {16,32,64}` — encoder latent d (ablation; 32 was Stage 2 best)
- `--lstm_hidden`, `--lstm_layers`, `--lstm_dropout`
- `--prev_embed_dim` — Variant B: previous-frame embedding width (default 64)
- `--teacher_forcing_ratio` — Variant B: 1.0 = pure teacher forcing, <1.0 = scheduled sampling
- `--pos_weight 6.24` — balance the ~13.8% foreground (default: vanilla BCE,
  matching Stage 2 for an apples-to-apples comparison)
- `--threshold` — sigmoid binarization threshold (default 0.5)
- `--patience`, `--lr_patience`, `--lr`, `--weight_decay`, `--epochs`, `--batch_size`

## Metrics

Static metrics (3.6.2) use `metrics.py` unchanged. Variant B flattens its
`(N,T,W,W)` predictions to `(N*T,W,W)` → **frame-averaged** IoU/Dice (IoU_seq,
3.6.3). **Temporal Consistency** (3.6.3) — `TC = (1/(T-1)) Σ_t ‖Ŷ_t−Ŷ_{t-1}‖`,
lower = smoother — is computed in `evaluate_stage3.py` so `metrics.py` stays
byte-identical to Stage 2.

## Variant B: autoregressive decoder + teacher forcing


\- **Encoder LSTM** (bidirectional) produces a per-step context `h_t` that has
  seen the whole CSI window.  
\- **Separate decoder** `AutoregressiveMaskDecoder` predicts each frame from
  `(h_t, Ŷ_{t-1})`: the previous mask is embedded (`--prev_embed_dim`, default
  64) and concatenated with `h_t` before the same MLP body (256→512→W²) used in
  Stage 2. The first step is seeded with a zero "start" frame.  
\- **Teacher forcing** (`--teacher_forcing_ratio`, default 1.0): during training
  the ground-truth `Y_{t-1}` is fed in with that probability; otherwise the
  model's own binarized, detached prediction is used (scheduled sampling).
  During evaluation the model always uses its own prediction, so val/test
  numbers reflect true autoregressive inference.  

This is a different model from Variant A, not a controlled ablation
of it — report it as such. Variant A vs Stage 2 isolates the LSTM; Variant B
demonstrates coherent sequence reconstruction (IoU_seq + TC), which is the
point of 3.5's seq-to-seq formulation.

## Outputs of a run

```
runs/<name>/
├── config.json                    # all hyperparameters + resolved dirs + N + n_params
├── norm_stats.json                # train-set μ, σ (reused at evaluation)
├── train_log.csv                  # per-epoch loss + val metrics
├── best.pt / last.pt              # checkpoints (best = highest val IoU)
├── test_metrics.json              # test metrics of best checkpoint (end of training)
├── metrics_<split>.json           # (evaluate) + temporal_consistency for B
├── per_sample_metrics_<split>.csv
└── qual_grid_<split>.png          # A: CSI|GT|Pred ; B: GT-seq over Pred-seq
```

## dataset_seq.py

PyTorch Dataset wrapper for the Stage 3 sequence datasets (3.3.3).

&nbsp;&nbsp;Variant A  (seq → 1 mask):  
&nbsp;&nbsp;&nbsp;&nbsp;X_variant_a.npy : (N, T, 64, 52) float32  — CSI windows  
&nbsp;&nbsp;&nbsp;&nbsp;Y_variant_a.npy : (N, 32, 32)    uint8     — mask of the last frame (idx 9)  

&nbsp;&nbsp;Variant B  (seq → seq):  
&nbsp;&nbsp;&nbsp;&nbsp;X_seq.npy : (N, T, 64, 52) float32  — CSI windows  (== X_variant_a)  
&nbsp;&nbsp;&nbsp;&nbsp;Y_seq.npy : (N, T, 32, 32) uint8     — one mask per time step  

&nbsp;&nbsp;seq_{train,val,test}_idx.npy — temporal block-based split indices.

### Normalization
Identical scheme to Stage 2: a single global z-score using statistics
computed on the TRAIN SPLIT ONLY, saved to `norm_stats.json` and reused at
evaluation. The CSI windows here carry an extra time axis, but the statistic
is still a pair of scalars (global mean, global std) over all elements, so
the same per-frame normalization that Stage 2 used applies unchanged to
every frame of every window.

## evaluate_stage3.py
Section 3.6 — Evaluating the Stage 3 Temporal Model
Loads the best checkpoint from a training run and reports test-set metrics,
a qualitative grid, and a per-sample CSV. Mirrors evaluate_stage2.py.

Static metrics (3.6.2) come from metrics.py UNCHANGED. For Variant B the
(N,T,W,W) predictions are flattened to (N*T,W,W) → frame-averaged IoU/Dice
(IoU_seq). Temporal Consistency (3.6.3) is computed HERE, on the binarized
sequence, so metrics.py stays byte-identical to Stage 2:

&nbsp;&nbsp;&nbsp;&nbsp;`TC = (1 / (T-1)) Σ_t ‖ Ŷ_t − Ŷ_{t-1} ‖   (mean abs diff per pixel)`

### Usage
    python evaluate_stage3.py \
      --data_dir /mnt/user-data/uploads \
      --run_dir  ./runs/stage3_a_d32 \
      --split    test --n_examples 12


## metrics.py
Section 3.6.2 — Metrics for Static Reconstruction
Pixel-wise and shape-overlap metrics for binary silhouette maps.

All metrics accept (B, W, W) tensors and return per-batch means.
Binary metrics (acc, IoU, Dice) require thresholding; MSE uses raw probas.

### Notation
&nbsp;&nbsp;&nbsp;&nbsp;y_true       : (B, W, W) {0,1} ground-truth mask  
&nbsp;&nbsp;&nbsp;&nbsp;y_pred_proba : (B, W, W) [0,1] predicted probability map  
&nbsp;&nbsp;&nbsp;&nbsp;y_pred_bin   : (B, W, W) {0,1} thresholded prediction at τ (default 0.5)  

NOTE: This file is reused UNCHANGED from Stage 2. Stage 3 Variant B
(sequence output) flattens its (N, T, W, W) tensors to (N*T, W, W) before
calling these functions, which yields frame-averaged metrics (IoU_seq in 3.6.3). Temporal Consistency (3.6.3) is computed in evaluate_stage3.py so
this module stays identical to the Stage 2 version.

## model_stage3.py
Section 3.5 — Stage 3: Temporal Encoder–LSTM–Decoder Reconstruction
Extends the Stage 2 static model with an LSTM that operates on the
sequence of per-frame CSI embeddings, so any IoU gain over Stage 2 is
attributable to temporal context.

&nbsp;&nbsp;&nbsp;&nbsp;per frame :  `e_t = E_θ(H_t),  e_t ∈ R^d`                  (CNN, reused)  
&nbsp;&nbsp;&nbsp;&nbsp;sequence  :  `h_t = LSTM(e_t, h_{t-1})`                    (NEW)  
&nbsp;&nbsp;&nbsp;&nbsp;Variant A :  `Ŷ_T = σ(D_ϕ(h_T) ∈ [0,1]^(W×W)`              (seq → 1 mask)  
&nbsp;&nbsp;&nbsp;&nbsp;Variant B :  `Ŷ_t = σ(D_ϕ^AR(h_t, Ŷ_{t-1}))  for t=1..T`   (seq → seq)  

### What is reused vs. new, per variant
&nbsp;&nbsp;Variant A  — the "isolate the LSTM" experiment (Experiment 2, 3.6.4):  
&nbsp;&nbsp;&nbsp;&nbsp;REUSED : CSIEncoder (E_θ), MaskDecoder (D_ϕ).  
&nbsp;&nbsp;&nbsp;&nbsp;NEW                           : the LSTM block only.  

The MaskDecoder is identical to Stage 2; only the width of the latent it consumes changes (Stage 2: d; here: the LSTM output width H). For the strictest controlled comparison, run Variant A with `--lstm_hidden == --latent_dim` so the decoder is dimensionally identical to Stage 2's and the LSTM is the sole architectural difference.  

&nbsp;&nbsp;Variant B  — the 3.5 sequence-to-sequence model:  
&nbsp;&nbsp;&nbsp;&nbsp;REUSED   : CSIEncoder (E_θ).  
&nbsp;&nbsp;&nbsp;&nbsp;NEW      : the (bidirectional) LSTM + a SEPARATE autoregressive decoder `D_ϕ^AR` that conditions each frame on the previous frame, `Ŷ_t = σ(D_ϕ^AR(h_t, Ŷ_{t-1}))`, trained with teacher forcing.  

Variant B: implementing the methodology's autoregressive decoder requires augmenting the decoder input with `Ŷ_{t-1}`, so Variant B carries its own decoder. The `Ŷ_{t-1}` feedback is what enforces output-space temporal coherence (the quantity the TC metric in 3.6.3 reports); a bidirectional encoder feeding an autoregressive output decoder is the standard seq2seq arrangement.

## train_stage3.py
Section 3.5 — Training the Stage 3 Temporal Encoder–LSTM–Decoder
Mirrors train_stage2.py so results are directly comparable.

  - Loss:        BCEWithLogitsLoss (numerically stable BCE)
                 optional --pos_weight for the ~14% foreground imbalance
                 Variant A → loss on the single last-frame mask
                 Variant B → mean over all T steps   (= L_seq, Eq. seq_loss)
  - Optimizer:   Adam with weight decay
  - Scheduler:   ReduceLROnPlateau on validation IoU (frame-averaged for B)
  - Early stop:  --patience epochs without val-IoU improvement
  - Checkpoint:  best (highest val IoU) and last
  - Log:         per-epoch CSV with train/val loss + all val metrics
  - Encoder:     trained FROM SCRATCH by default
                 (--init_encoder_from <stage2 best.pt> to warm-start; ablation)

### Usage
```
  # Variant A — unidirectional, seq → 1 mask  (default)
  python train_stage3.py --variant a \
      --data_dir /mnt/user-data/uploads --out_dir ./runs/stage3_a_d32 \
      --latent_dim 32 --lstm_hidden 64 --epochs 100 --batch_size 32 --lr 1e-3

  # Variant B — bidirectional, seq → seq
  python train_stage3.py --variant b \
      --data_dir /mnt/user-data/uploads --out_dir ./runs/stage3_b_d32 \
      --latent_dim 32 --lstm_hidden 64

  # Strict controlled comparison vs Stage 2 (decoder dimensionally identical)
  python train_stage3.py --variant a --lstm_hidden 32
```

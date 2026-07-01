import argparse
import json
import os
from datetime import datetime
import numpy as np

def build_d_frame(X_csi, y_img_ids, Y_masks, mask_ids):
    """
    Frame-level dataset — Stage 2 (CNN + MLP, no temporal modelling).
      Encoder:  z   = E_θ(X_t)        with X_t ∈ ℝ^(T×F)
      Decoder:  Ŷ_t = σ(D_ϕ(z))       with Ŷ_t ∈ [0,1]^(W×W)
      Target:   Y_t (mask paired with X_t via image ID)
    Returns
    -------
    X_frame : (N, T, F) float32
    Y_frame : (N, W, W) uint8 in {0,1}
    img_ids : (N,) int64  — provenance tracking
    stats   : dict        — construction quality metrics
    """    
    mask_id_to_idx = {int(mid): i for i, mid in enumerate(mask_ids)}

    N = len(X_csi)
    W = Y_masks.shape[1]
    Y_frame = np.zeros((N, W, W), dtype=np.uint8)

    matched, missing = 0, []
    for i, img_id in enumerate(y_img_ids):
        img_id_int = int(img_id)
        if img_id_int in mask_id_to_idx:
            Y_frame[i] = Y_masks[mask_id_to_idx[img_id_int]]
            matched += 1
        else:
            missing.append(img_id_int)

    fg = Y_frame.reshape(N, -1).mean(axis=1)
    stats = {
        "N": int(N),
        "matched": int(matched),
        "missing": int(len(missing)),
        "missing_ids_sample": missing[:20],
        "fg_ratio_mean": float(fg.mean()),
        "fg_ratio_std":  float(fg.std()),
        "fg_ratio_min":  float(fg.min()),
        "fg_ratio_max":  float(fg.max()),
        "zero_fg_count": int((fg == 0).sum()),
    }
    return X_csi.copy(), Y_frame, np.asarray(y_img_ids).copy(), stats


def build_d_variant_a(X_frame, Y_frame, img_ids, seq_len, stride):
    """
    Many-to-one dataset — Stage 3 Variant A (CNN + LSTM → single mask).
      Encoder:  z_t  = E_θ(X_t)                t = 1..T_seq
      LSTM:     h_t  = LSTM(z_t, h_{t-1})
      Decoder:  Ŷ    = σ(D_ϕ(h_{T_seq}))       last hidden state only
      Target:   Y_{T_seq} (mask of the LAST CSI tensor in the window)
    Why last (not center)?
      A unidirectional LSTM's final hidden state h_T encodes context
      from [X_1..X_T]. The natural prediction target is Y_T.
      For center-frame targeting you would need a BiLSTM.
    Returns
    -------
    X_seq      : (N_seq, seq_len, T, F) float32
    Y_single   : (N_seq, W, W)          uint8 — last-frame mask
    target_ids : (N_seq,)               int64 — last-frame image IDs
    target_idx : int                          — position in window (= seq_len-1)
    """
    N = len(X_frame)
    n_seqs = (N - seq_len) // stride + 1
    target_idx = seq_len - 1

    X_seq      = np.zeros((n_seqs, seq_len, *X_frame.shape[1:]), dtype=X_frame.dtype)
    Y_single   = np.zeros((n_seqs, *Y_frame.shape[1:]),          dtype=Y_frame.dtype)
    target_ids = np.zeros(n_seqs, dtype=img_ids.dtype)

    for i in range(n_seqs):
        start = i * stride
        X_seq[i]      = X_frame[start : start + seq_len]
        Y_single[i]   = Y_frame[start + target_idx]
        target_ids[i] = img_ids[start + target_idx]

    return X_seq, Y_single, target_ids, target_idx


def build_d_seq(X_frame, Y_frame, img_ids, seq_len, stride):
    """
    Many-to-many dataset — Stage 3 Variant B (CNN + LSTM → sequence of masks).
      Encoder:  z_t  = E_θ(X_t)                 t = 1..T_seq
      LSTM:     h_t  = LSTM(z_t, h_{t-1})
      Decoder:  Ŷ_t  = σ(D_ϕ(h_t))              one mask per time step
      Target:   {Y_t}_{t=1..T_seq}              one mask per CSI tensor
    Returns
    -------
    X_seq   : (N_seq, seq_len, T, F) float32
    Y_seq   : (N_seq, seq_len, W, W) uint8
    seq_ids : (N_seq, seq_len)       int64 — image IDs per position
    """
    N = len(X_frame)
    n_seqs = (N - seq_len) // stride + 1

    X_seq   = np.zeros((n_seqs, seq_len, *X_frame.shape[1:]), dtype=X_frame.dtype)
    Y_seq   = np.zeros((n_seqs, seq_len, *Y_frame.shape[1:]), dtype=Y_frame.dtype)
    seq_ids = np.zeros((n_seqs, seq_len), dtype=img_ids.dtype)

    for i in range(n_seqs):
        start = i * stride
        end   = start + seq_len
        X_seq[i]   = X_frame[start:end]
        Y_seq[i]   = Y_frame[start:end]
        seq_ids[i] = img_ids[start:end]

    return X_seq, Y_seq, seq_ids


def split_temporal_blocks(N, train_frac=0.7, val_frac=0.15, gap=None, seq_len=10):
    """
    Generate train / val / test index ranges using contiguous temporal blocks
    rather than random sampling. Random splitting on overlapping sequence data
    causes leakage (consecutive sequences share seq_len-1 of seq_len tensors).
 
    A gap is inserted between splits so that no sequence in one split
    overlaps temporally with a sequence in another.
    Parameters
    ----------
    N          : total number of samples
    train_frac : fraction for training
    val_frac   : fraction for validation  (rest goes to test)
    gap        : number of indices to skip between splits (defaults to seq_len)
    seq_len    : sequence length (used only for the default gap)
    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int
    """
    if gap is None:
        gap = seq_len

    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    train_idx = np.arange(0,                   n_train - gap)
    val_idx   = np.arange(n_train,             n_train + n_val - gap)
    test_idx  = np.arange(n_train + n_val,     N)
    return train_idx, val_idx, test_idx


# ── CLI entry point ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Section 3.3.3 — temporal alignment and dataset construction"
    )
    p.add_argument("--csi_tensors",    required=True, help="X_csi_tensors.npy from 3.3.1")
    p.add_argument("--csi_image_ids",  required=True, help="y_image_ids.npy from 3.3.1")
    p.add_argument("--masks",          required=True, help="Y_masks.npy from 3.3.2")
    p.add_argument("--mask_image_ids", required=True, help="image_ids.npy from 3.3.2")
    p.add_argument("--output_dir",     default="./datasets")
    p.add_argument("--seq_len",        type=int, default=10, help="Sequence length for D_variant_a and D_seq")
    p.add_argument("--stride",         type=int, default=1,  help="Sliding window stride (1 = max overlap)")
    p.add_argument("--write_splits",   action="store_true",  help="Also write train/val/test index files")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ──
    print("Loading inputs...")
    X_csi      = np.load(args.csi_tensors)
    y_img_ids  = np.asarray(np.load(args.csi_image_ids, allow_pickle=True))
    Y_masks    = np.load(args.masks)
    mask_ids   = np.asarray(np.load(args.mask_image_ids, allow_pickle=True))

    print(f"  X_csi_tensors : {X_csi.shape}     {X_csi.dtype}")
    print(f"  y_image_ids   : {y_img_ids.shape}        {y_img_ids.dtype}")
    print(f"  Y_masks       : {Y_masks.shape} {Y_masks.dtype}")
    print(f"  image_ids     : {mask_ids.shape}      {mask_ids.dtype}")

    # ── D_frame ──
    print("\n" + "─" * 60)
    print("Building D_frame (Stage 2)...")
    X_f, Y_f, ids_f, stats_f = build_d_frame(X_csi, y_img_ids, Y_masks, mask_ids)
    np.save(os.path.join(args.output_dir, "X_frame.npy"),          X_f)
    np.save(os.path.join(args.output_dir, "Y_frame.npy"),          Y_f)
    np.save(os.path.join(args.output_dir, "frame_image_ids.npy"),  ids_f)
    print(f"  Matched: {stats_f['matched']} / {stats_f['N']}  (missing: {stats_f['missing']})")
    print(f"  FG ratio: {stats_f['fg_ratio_mean']:.4f} ± {stats_f['fg_ratio_std']:.4f}")
    print(f"  X_frame: {X_f.shape}    Y_frame: {Y_f.shape}")

    # ── D_variant_a ──
    print("\n" + "─" * 60)
    print(f"Building D_variant_a (Stage 3A, seq_len={args.seq_len}, stride={args.stride})...")
    X_va, Y_va, ids_va, target_idx = build_d_variant_a(X_f, Y_f, ids_f, args.seq_len, args.stride)
    np.save(os.path.join(args.output_dir, "X_variant_a.npy"),          X_va)
    np.save(os.path.join(args.output_dir, "Y_variant_a.npy"),          Y_va)
    np.save(os.path.join(args.output_dir, "variant_a_target_ids.npy"), ids_va)
    print(f"  Sequences: {len(X_va)}")
    print(f"  Target index in window: {target_idx} (last frame)")
    print(f"  X_variant_a: {X_va.shape}    Y_variant_a: {Y_va.shape}")

    # ── D_seq ──
    print("\n" + "─" * 60)
    print(f"Building D_seq (Stage 3B, seq_len={args.seq_len}, stride={args.stride})...")
    X_s, Y_s, ids_s = build_d_seq(X_f, Y_f, ids_f, args.seq_len, args.stride)
    np.save(os.path.join(args.output_dir, "X_seq.npy"),         X_s)
    np.save(os.path.join(args.output_dir, "Y_seq.npy"),         Y_s)
    np.save(os.path.join(args.output_dir, "seq_image_ids.npy"), ids_s)
    print(f"  Sequences: {len(X_s)}")
    print(f"  X_seq: {X_s.shape}    Y_seq: {Y_s.shape}")

    # ── Splits (optional) ──
    if args.write_splits:
        print("\n" + "─" * 60)
        print("Writing temporal block-based train/val/test splits...")
        tr_f, va_f, te_f = split_temporal_blocks(len(X_f), seq_len=1)
        np.save(os.path.join(args.output_dir, "frame_train_idx.npy"), tr_f)
        np.save(os.path.join(args.output_dir, "frame_val_idx.npy"),   va_f)
        np.save(os.path.join(args.output_dir, "frame_test_idx.npy"),  te_f)
        print(f"  D_frame:    train={len(tr_f)}  val={len(va_f)}  test={len(te_f)}")

        tr_s, va_s, te_s = split_temporal_blocks(len(X_s), seq_len=args.seq_len)
        np.save(os.path.join(args.output_dir, "seq_train_idx.npy"),   tr_s)
        np.save(os.path.join(args.output_dir, "seq_val_idx.npy"),     va_s)
        np.save(os.path.join(args.output_dir, "seq_test_idx.npy"),    te_s)
        print(f"  Sequences:  train={len(tr_s)}  val={len(va_s)}  test={len(te_s)}")
        print(f"  (gap = {args.seq_len} between splits to prevent overlap)")

    # ── Metadata ──
    metadata = {
        "created": datetime.now().isoformat(),
        "section": "3.3.3 — Temporal Alignment and Dataset Construction",
        "seq_len": args.seq_len,
        "stride": args.stride,
        "d_frame": {
            "X_shape": list(X_f.shape),
            "Y_shape": list(Y_f.shape),
            "stage": "Stage 2 (CNN + MLP)",
            "stats": {k: v for k, v in stats_f.items() if k != "missing_ids_sample"},
        },
        "d_variant_a": {
            "X_shape": list(X_va.shape),
            "Y_shape": list(Y_va.shape),
            "target_idx_in_window": target_idx,
            "stage": "Stage 3 Variant A (CNN + LSTM → 1 mask via last hidden state)",
        },
        "d_seq": {
            "X_shape": list(X_s.shape),
            "Y_shape": list(Y_s.shape),
            "stage": "Stage 3 Variant B (CNN + LSTM → mask per time step)",
        },
        "notes": {
            "stride_warning": (
                f"stride={args.stride} produces overlapping sequences sharing "
                f"{args.seq_len - args.stride} of {args.seq_len} CSI tensors. "
                "Use temporal block-based splitting (--write_splits) to prevent leakage."
            ),
            "within_tensor_dynamics": (
                "The CNN encoder compresses 64 frames of intra-second dynamics "
                "into a single z. The LSTM operates only at the inter-tensor "
                "(~1 Hz) timescale."
            ),
        },
    }
    with open(os.path.join(args.output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "═" * 60)
    print(f"DONE. All outputs in: {args.output_dir}")
    print("═" * 60)


if __name__ == "__main__":
    main()

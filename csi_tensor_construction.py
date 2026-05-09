import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

#indices 6–31 and 33–58 of the 128 raw slots carry L-LTF data.
CSI_VALID_SUBCARRIER_INDEX = list(range(6, 32)) + list(range(33, 59))
S = len(CSI_VALID_SUBCARRIER_INDEX)  # 52 subcarriers


def load_csi_data(csi_csv_path: str, csi_npy_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    #load metadata from CSV (only need id and sensor_timestamp columns)
    df = pd.read_csv(csi_csv_path, usecols=["id", "sensor_timestamp"])
    image_ids = df["id"].values.astype(int)
    timestamps = df["sensor_timestamp"].values.astype(float)

    #load pre-parsed complex CSI and extract amplitude
    csi_complex = np.load(csi_npy_path) #(N, 52) complex64
    amplitudes = np.abs(csi_complex).astype(np.float32) #(N, 52) float32

    #assert len(amplitudes) == len(timestamps) == len(image_ids), \
    #    "Mismatch between CSI data and CSV metadata lengths."
    
    print(amplitudes[0,:])

    #plot spectogram of first 100 packets to verify data loading
    amplitudes100 = amplitudes[:100]

    plt.figure(figsize=(10, 6))
    plt.imshow(amplitudes100.T, aspect='auto', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title('CSI Amplitude Spectrogram')
    plt.xlabel('Packet Index')
    plt.ylabel('Subcarrier Index')
    plt.show()

    print(f"[INFO] Loaded {len(amplitudes)} CSI packets, "
          f"S = {amplitudes.shape[1]} subcarriers, "
          f"duration = {timestamps[-1] - timestamps[0]:.2f}s")

    return amplitudes, timestamps, image_ids


def compute_window_boundaries(
    timestamps: np.ndarray,
    window_sec: float = 1.0,
    stride_sec: float = 0.5
) -> List[Tuple[float, float]]:
    
    t_start = timestamps[0]
    t_end = timestamps[-1]

    windows = []
    w_start = t_start
    while w_start + window_sec <= t_end:
        windows.append((w_start, w_start + window_sec))
        w_start += stride_sec

    print(f"[INFO] Created {len(windows)} windows "
          f"(duration={window_sec}s, stride={stride_sec}s)")

    return windows


def bin_packets(
    amplitudes: np.ndarray,
    timestamps: np.ndarray,
    window_start: float,
    window_end: float,
    T: int,
    aggregation: str = "mean",
    imputation: str = "linear"
) -> np.ndarray:
    
    bin_duration = (window_end - window_start) / T

    #select packets falling within this window
    mask = (timestamps >= window_start) & (timestamps < window_end)
    window_ts = timestamps[mask]
    window_amp = amplitudes[mask]  #(n_packets, S)

    #assign each packet to a temporal bin
    relative_ts = window_ts - window_start
    bin_indices = np.minimum((relative_ts / bin_duration).astype(int), T - 1)

    #aggregate packets into bins
    tensor = np.full((T, S), np.nan, dtype=np.float32)  #NaN marks empty bins

    for b in range(T):
        packets_in_bin = window_amp[bin_indices == b]

        if len(packets_in_bin) == 0:
            continue  #leave as NaN, will be imputed later
        elif len(packets_in_bin) == 1:
            tensor[b] = packets_in_bin[0] #if only one packet in this bin, take it as is
        else:
            #if multiple packets in this bin, to be aggregated
            if aggregation == "mean":
                tensor[b] = np.mean(packets_in_bin, axis=0)
            elif aggregation == "max":
                tensor[b] = np.max(packets_in_bin, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    #impute empty bins
    empty_mask = np.isnan(tensor[:, 0])  #check first subcarrier for emptiness
    n_empty = empty_mask.sum()

    if n_empty > 0 and n_empty < T:
        if imputation == "zero":
            tensor[empty_mask] = 0.0

        elif imputation == "linear":
            #interpolate each subcarrier across time bins
            valid_bins = np.where(~empty_mask)[0]
            empty_bins = np.where(empty_mask)[0]
            for s in range(S):
                tensor[empty_bins, s] = np.interp(
                    empty_bins, valid_bins, tensor[valid_bins, s]
                )

        elif imputation == "nearest":
            #for each empty bin, copy the nearest non-empty bin
            valid_bins = np.where(~empty_mask)[0]
            empty_bins = np.where(empty_mask)[0]
            for eb in empty_bins:
                nearest_idx = valid_bins[np.argmin(np.abs(valid_bins - eb))]
                tensor[eb] = tensor[nearest_idx]

        else:
            raise ValueError(f"Unknown imputation: {imputation}")

    elif n_empty == T:
        #entire window is empty (rare edge case), fill with zeros or leave as NaN based on imputation method
        tensor[:] = 0.0

    return tensor


def get_window_image_id(
    image_ids: np.ndarray,
    timestamps: np.ndarray,
    window_start: float,
    window_end: float,
    image_dir: str,
    random_sample: bool = False
) -> Optional[int]:
    
    mask = (timestamps >= window_start) & (timestamps < window_end)
    window_ids = image_ids[mask]
    window_ts = timestamps[mask]

    if len(window_ids) == 0:
        return None

    #get unique IDs and verify the image files exist
    unique_ids = []
    for uid in np.unique(window_ids):
        img_path = os.path.join(image_dir, f"{uid}.png")
        if os.path.isfile(img_path):
            unique_ids.append(uid)

    if len(unique_ids) == 0:
        return None

    if random_sample:
        return int(np.random.choice(unique_ids))
    else:
        #pick the image closest to the window's temporal centre
        window_centre = (window_start + window_end) / 2.0
        centre_idx = np.argmin(np.abs(window_ts - window_centre))
        centre_id = int(window_ids[centre_idx])

        #fall back if that specific file doesn't exist
        if centre_id in unique_ids:
            return centre_id
        else:
            return unique_ids[len(unique_ids) // 2]


def build_dataset(
    csi_csv_path: str,
    csi_npy_path: str,
    image_dir: str,
    output_dir: str,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    T: int = 64,
    aggregation: str = "mean",
    imputation: str = "linear"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    

    amplitudes, timestamps, image_ids = load_csi_data(csi_csv_path, csi_npy_path)

    windows = compute_window_boundaries(timestamps, window_sec, stride_sec)

    print(f"\n[INFO] Processing {len(windows)} windows to build dataset...")
    for window in windows[:5]:
        #print values for first 5 windows to verify correct windowing
        print(f"Window: {window[0]:.2f} to {window[1]:.2f}")


    X_list = []
    y_ids = []
    empty_bin_counts = []
    for i, (w_start, w_end) in enumerate(windows):
        #build the CSI tensor for this window
        tensor = bin_packets(
            amplitudes, timestamps, w_start, w_end,
            T=T, aggregation=aggregation, imputation=imputation
        )
        X_list.append(tensor)

        img_id = get_window_image_id(
            image_ids, timestamps, w_start, w_end,
            image_dir, random_sample=False
        )
        y_ids.append(img_id if img_id is not None else -1)

    print(f"[INFO] Built dataset with {len(X_list)} samples.")
    print(X_list[0])

    X = np.stack(X_list, axis=0)  #(N_windows, T, S)
    y_ids = np.array(y_ids)

    #remove windows with no valid image
    valid_mask = y_ids >= 0
    X = X[valid_mask]
    y_ids = y_ids[valid_mask]

    #save outputs
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_csi_tensors.npy"), X)
    np.save(os.path.join(output_dir, "y_image_ids.npy"), y_ids)

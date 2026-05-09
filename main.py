import argparse
from csi_tensor_construction import build_dataset

#run the script with:
# & C:\Users\mohammed.zanella\.conda\envs\hwF21MP\python.exe "c:/Users/mohammed.zanella/OneDrive - Heriot-Watt University/F21MP/Project/Wifi-CSI-Imaging-DeepLearningModel/main.py" --csi_csv Dataset/csi.csv --csi_npy Dataset/csiComplex.npy --image_dir "Dataset/640" --T 64 --window_sec 1.0 --stride_sec 0.5 --aggregation mean --imputation linear

#cli entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Section 3.3.1 – CSI Tensor Construction"
    )
    parser.add_argument("--csi_csv", type=str, required=True,
                        help="Path to csi.csv")
    parser.add_argument("--csi_npy", type=str, required=True,
                        help="Path to csiComplex.npy")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to folder containing PNG images")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Where to save the constructed tensors")
    parser.add_argument("--window_sec", type=float, default=1.0,
                        help="Window duration in seconds (default: 1.0)")
    parser.add_argument("--stride_sec", type=float, default=0.5,
                        help="Stride between windows in seconds (default: 0.5)")
    parser.add_argument("--T", type=int, default=64,
                        help="Number of temporal bins per window (default: 64)")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "max"],
                        help="Aggregation for bins with multiple packets")
    parser.add_argument("--imputation", type=str, default="linear",
                        choices=["zero", "linear", "nearest"],
                        help="Imputation for empty bins")

    args = parser.parse_args()

    build_dataset(
        csi_csv_path=args.csi_csv,
        csi_npy_path=args.csi_npy,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        T=args.T,
        aggregation=args.aggregation,
        imputation=args.imputation,
    )
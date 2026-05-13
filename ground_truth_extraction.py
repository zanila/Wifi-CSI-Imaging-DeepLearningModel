import argparse
import json
import os
import sys

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# & C:\Users\mohammed.zanella\.conda\envs\hwF21MP\python.exe "c:/Users/mohammed.zanella/OneDrive - Heriot-Watt University/F21MP/Project/Wifi-CSI-Imaging-DeepLearningModel/ground_truth_extraction.py" --image_dir "Dataset/Sample Frames/" --output_dir "gt_output_temp/" --mask_size 32

def load_yolo_model(model_name: str):
    model = YOLO(model_name)
    print(f"[INFO] Loaded YOLO model: {model_name}")
    return model


def get_person_mask_and_bbox(model, img_path: str, confidence_threshold: float):
    
    results = model(img_path, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None, None, None

    #filter to 'person' class (COCO class 0) above confidence threshold
    person_indices = []
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0 and conf >= confidence_threshold:  # 0 = 'person' in COCO
            person_indices.append((i, conf))

    if not person_indices:
        return None, None, None

    #pick the highest-confidence person detection
    best_idx, best_conf = max(person_indices, key=lambda x: x[1])

    #extract segmentation mask (H×W binary from YOLO)
    mask = r.masks[best_idx].data[0].cpu().numpy()  # (model_H, model_W)

    #resize mask to original image dimensions if needed
    img_h, img_w = r.orig_shape
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    #extract bounding box [x1, y1, x2, y2]
    bbox = r.boxes[best_idx].xyxy[0].cpu().numpy().astype(int)

    return mask, bbox, best_conf


def make_square_bbox(x1, y1, x2, y2, img_h, img_w, padding_ratio=0.1):
    
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    #square side = max dimension + padding
    side = max(bw, bh)
    side = int(side * (1.0 + 2 * padding_ratio))

    half = side / 2.0
    x1_new = max(0, int(cx - half))
    y1_new = max(0, int(cy - half))
    x2_new = min(img_w, int(cx + half))
    y2_new = min(img_h, int(cy + half))

    return x1_new, y1_new, x2_new, y2_new


def process_single_image(model, img_path: str, mask_size: int,
                         bbox_padding: float, confidence_threshold: float):
   
    img = cv2.imread(img_path)
    if img is None:
        return None, {"status": "read_error"}

    img_h, img_w = img.shape[:2]

    #step 1: YOLO instance segmentation
    seg_mask, bbox, conf = get_person_mask_and_bbox(
        model, img_path, confidence_threshold
    )

    if seg_mask is None:
        return None, {"status": "no_person_detected"}

    x1, y1, x2, y2 = bbox

    #step 2: Make square bounding box
    sx1, sy1, sx2, sy2 = make_square_bbox(
        x1, y1, x2, y2, img_h, img_w, bbox_padding
    )

    #step 3: Crop mask to square bbox
    cropped_mask = seg_mask[sy1:sy2, sx1:sx2]

    #step 4: Resize to W×W
    resized = cv2.resize(
        cropped_mask.astype(np.float32),
        (mask_size, mask_size),
        interpolation=cv2.INTER_AREA  # anti-aliased downscaling
    )

    #step 5: Threshold to binary {0, 1}
    binary_mask = (resized > 0.5).astype(np.uint8)

    #compute metadata
    fg_pixels = int(binary_mask.sum())
    total_pixels = mask_size * mask_size
    fg_ratio = fg_pixels / total_pixels

    metadata = {
        "status": "ok",
        "confidence": round(float(conf), 4),
        "original_bbox": [int(x1), int(y1), int(x2), int(y2)],
        "square_bbox": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "fg_pixels": fg_pixels,
        "fg_ratio": round(fg_ratio, 4),
    }

    return binary_mask, metadata

def create_qa_grid(image_dir: str, image_ids: list, masks: np.ndarray,
                   metadata_list: list, output_path: str, max_cols=8, max_rows=6):
    
    from PIL import Image, ImageDraw, ImageFont

    n_show = min(len(image_ids), max_cols * max_rows)
    #Sample evenly across the dataset
    indices = np.linspace(0, len(image_ids) - 1, n_show, dtype=int)

    n_cols = min(n_show, max_cols)
    n_rows = int(np.ceil(n_show / n_cols))

    cell_w, cell_h = 180, 200  #each cell: image on top, mask below
    canvas_w = n_cols * cell_w
    canvas_h = n_rows * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    for pos, idx in enumerate(indices):
        col = pos % n_cols
        row = pos // n_cols
        x_off = col * cell_w
        y_off = row * cell_h

        img_id = image_ids[idx]

        #find the image file
        img_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue

        #original image thumbnail
        orig = Image.open(img_path).resize((cell_w - 10, 90))
        canvas.paste(orig, (x_off + 5, y_off + 5))

        #32×32 mask upscaled
        m = masks[idx]
        mask_img = Image.fromarray((m * 255).astype(np.uint8), mode='L')
        mask_up = mask_img.resize((90, 90), Image.NEAREST)
        mask_rgb = Image.merge('RGB', [mask_up, mask_up, mask_up])
        canvas.paste(mask_rgb, (x_off + 45, y_off + 100))

        #Label
        meta = metadata_list[idx]
        label = f"ID:{img_id} fg:{meta.get('fg_ratio', 0):.0%}"
        draw.text((x_off + 5, y_off + 192), label, fill=(200, 200, 200))

    canvas.save(output_path)
    print(f"[INFO] QA grid saved: {output_path} ({n_show} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="3.3.2 Ground-Truth Extraction: RGB frames → binary silhouette masks"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing PNG images (e.g., /data/wificam/j3/640/)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save outputs (Y_masks.npy, image_ids.npy, etc.)"
    )
    parser.add_argument(
        "--mask_size", type=int, default=32,
        help="Output mask resolution W×W (default: 32)"
    )
    parser.add_argument(
        "--bbox_padding", type=float, default=0.1,
        help="Fractional padding around bounding box (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5,
        help="Minimum YOLO confidence for person detection (default: 0.5)"
    )
    parser.add_argument(
        "--yolo_model", type=str, default="yolov8n-seg.pt",
        help="YOLOv8 segmentation model (default: yolov8n-seg.pt)"
    )
    parser.add_argument(
        "--skip_qa", action="store_true",
        help="Skip generating QA visualization grid"
    )

    args = parser.parse_args()

    #validate input directory
    if not os.path.isdir(args.image_dir):
        print(f"[ERROR] Image directory not found: {args.image_dir}")
        sys.exit(1)

    #create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    #discover image files (numeric ID PNGs)
    image_files = []
    for f in os.listdir(args.image_dir):
        if f.endswith('.png'):
            stem = f.replace('.png', '')
            if stem.isdigit():
                image_files.append((int(stem), f))

    image_files.sort(key=lambda x: x[0])
    print(f"[INFO] Found {len(image_files)} PNG images in {args.image_dir}")

    if len(image_files) == 0:
        print("[ERROR] No PNG images found.")
        sys.exit(1)

    #load YOLO model
    model = load_yolo_model(args.yolo_model)

    #Process all images
    all_masks = []
    all_ids = []
    all_metadata = []
    skipped = 0

    for i, (img_id, filename) in enumerate(image_files):
        img_path = os.path.join(args.image_dir, filename)
        mask, meta = process_single_image(
            model, img_path, args.mask_size,
            args.bbox_padding, args.confidence_threshold
        )

        if mask is None:
            skipped += 1
            meta["image_id"] = img_id
            all_metadata.append(meta)
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_files)}] ID={img_id}: SKIPPED ({meta['status']})")
            continue

        all_masks.append(mask)
        all_ids.append(img_id)
        meta["image_id"] = img_id
        all_metadata.append(meta)

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{len(image_files)}] ID={img_id}: "
                  f"conf={meta['confidence']:.3f}, fg={meta['fg_ratio']:.1%}")

    #convert to numpy arrays
    Y_masks = np.array(all_masks, dtype=np.uint8)   # (N, W, W)
    image_ids = np.array(all_ids, dtype=np.int64)    # (N,)

    print(f"\n{'='*60}")
    print(f"[RESULTS]")
    print(f"  Total images:    {len(image_files)}")
    print(f"  Processed:       {len(all_masks)}")
    print(f"  Skipped:         {skipped}")
    print(f"  Y_masks shape:   {Y_masks.shape}")
    print(f"  Mask resolution: {args.mask_size}×{args.mask_size}")

    if len(all_masks) > 0:
        fg_ratios = [m["fg_ratio"] for m in all_metadata if m["status"] == "ok"]
        print(f"  Avg fg ratio:    {np.mean(fg_ratios):.1%}")
        print(f"  Fg ratio range:  [{np.min(fg_ratios):.1%}, {np.max(fg_ratios):.1%}]")

        confs = [m["confidence"] for m in all_metadata if m["status"] == "ok"]
        print(f"  Avg confidence:  {np.mean(confs):.3f}")
    print(f"{'='*60}")

    #save outputs
    masks_path = os.path.join(args.output_dir, "Y_masks.npy")
    ids_path = os.path.join(args.output_dir, "image_ids.npy")
    meta_path = os.path.join(args.output_dir, "mask_metadata.json")

    np.save(masks_path, Y_masks)
    np.save(ids_path, image_ids)

    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[SAVED]")
    print(f"  {masks_path}  ({Y_masks.nbytes / 1024:.1f} KB)")
    print(f"  {ids_path}")
    print(f"  {meta_path}")

    #generate QA grid
    if not args.skip_qa and len(all_masks) > 0:
        qa_path = os.path.join(args.output_dir, "qa_grid.png")
        ok_metadata = [m for m in all_metadata if m["status"] == "ok"]
        create_qa_grid(
            args.image_dir, all_ids, Y_masks,
            ok_metadata, qa_path
        )


if __name__ == "__main__":
    main()
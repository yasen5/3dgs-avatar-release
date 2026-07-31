#!/usr/bin/env python3
"""Generate person masks for a directory of frames using Sapiens2
(facebook/sapiens2-seg-0.4b, HF semantic segmentation). Mask = any non-
Background class. Writes 0/255 uint8 PNGs into --output, matching the
binary-mask convention MHRNativeDataset expects (mask != 0 == valid).

This is the default mask source for zju_377_mhr (see configs/dataset/
zju_377_mhr.yaml, mask_dir: masks) — it replaces the dataset's stock
CIHP/SCHP parsing masks, which were noisier around limbs/hair.

Example:
    python scripts/gen_sapiens2_masks.py \\
        --images /path/to/CoreView_377/Camera_B1 \\
        --output /path/to/CoreView_377/Camera_B1/masks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="facebook/sapiens2-seg-0.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overlay-dir", type=Path, default=None, help="optional debug overlays")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    print(f"Loading {args.model} ...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval().to(args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    if not image_paths:
        print(f"No images found in {args.images}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} frames from {args.images}", flush=True)
    for i, img_path in enumerate(image_paths):
        frame_id = img_path.stem
        out_path = args.output / f"{frame_id}.png"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[{frame_id}] failed to read, skipping", flush=True)
            continue
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[..., ::-1])

        inputs = processor(images=rgb, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]
        seg = seg.cpu().numpy()

        person_mask = (seg != 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_path), person_mask)

        if args.overlay_dir is not None and i % 40 == 0:
            overlay = bgr.copy()
            overlay[person_mask == 0] = (overlay[person_mask == 0] * 0.3).astype(np.uint8)
            cv2.imwrite(str(args.overlay_dir / f"{frame_id}.jpg"), overlay)

        if i % 50 == 0:
            print(f"[{i+1}/{len(image_paths)}] {frame_id}: mask area frac={person_mask.mean()/255:.4f}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

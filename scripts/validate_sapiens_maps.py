#!/usr/bin/env python3
"""Build full-frame Sapiens depth/normal visual validation contact sheets.

This script only reads already generated maps and images; it performs no model
inference.  Each output row is ``original | depth context | normal context``
for one representative frame, with the subject's complete source image kept
visible in every panel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 2**63 - 1, path.stem


def _depth_context(image: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = (mask > 0) & np.isfinite(depth)
    values = depth[valid]
    if values.size == 0:
        raise ValueError("depth map has no finite foreground values")
    lo, hi = np.percentile(values, [2.0, 98.0]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(image, 0.35, color, 0.65, 0.0)
    context = image.copy()
    context[valid] = blended[valid]
    return context


def _normal_context(image: np.ndarray, normal_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blended = cv2.addWeighted(image, 0.35, normal_bgr, 0.65, 0.0)
    context = image.copy()
    context[mask > 0] = blended[mask > 0]
    return context


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 38), (0, 0, 0), -1)
    cv2.putText(result, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=("bike", "citron", "jogging", "lab", "parkinglot", "seattle"))
    parser.add_argument("--samples-per-subject", type=int, default=3)
    parser.add_argument("--max-width", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_subject < 1 or args.max_width < 64:
        raise ValueError("samples-per-subject must be positive and max-width must be at least 64")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        root = (args.data_root / subject).expanduser().resolve()
        image_paths = sorted((root / "images").glob("*.png"), key=_numeric_key)
        if not image_paths:
            raise FileNotFoundError(f"no images found for {subject} under {root / 'images'}")
        sample_count = min(args.samples_per_subject, len(image_paths))
        indices = np.linspace(0, len(image_paths) - 1, sample_count, dtype=int).tolist()
        rows = []
        for row_index, image_index in enumerate(indices, start=1):
            image_path = image_paths[image_index]
            frame_id = image_path.stem
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(root / "masks" / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
            depth = np.load(root / "sapiens_depth" / f"{frame_id}.npy").astype(np.float32)
            normal = cv2.imread(str(root / "sapiens_normal" / f"{frame_id}.png"), cv2.IMREAD_COLOR)
            if image is None or mask is None or normal is None:
                raise RuntimeError(f"missing validation input for {subject}/{frame_id}")
            if image.shape[:2] != mask.shape or image.shape[:2] != depth.shape or image.shape[:2] != normal.shape[:2]:
                raise ValueError(f"full-frame shape mismatch for {subject}/{frame_id}")
            original = _label(image, f"{subject}/{frame_id} original")
            depth_panel = _label(_depth_context(image, depth, mask), "Sapiens depth in full image")
            normal_panel = _label(_normal_context(image, normal, mask), "Sapiens normals in full image")
            target_width = min(args.max_width, image.shape[1])
            target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
            panels = [cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA) for panel in (original, depth_panel, normal_panel)]
            row = cv2.hconcat(panels)
            cv2.putText(row, f"sample {row_index}/{sample_count}", (12, row.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(row)
        sheet = cv2.vconcat(rows)
        output_path = output_dir / f"{subject}_sapiens_context.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {output_path}")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

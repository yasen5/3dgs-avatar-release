"""Render NeuMan person masks over their paired source images.

Example:
    python scripts/render_neuman_mask_overlays.py \
        --root /mnt/ssd2/exavatar_neuman_combined/data/citron \
        --frames 0 12 24 36 \
        --output output/neuman_mask_overlays
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="NeuMan sequence directory")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame numbers to render")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--panel-width", type=int, default=638, help="Contact-sheet panel width")
    return parser.parse_args()


def load_pair(root: Path, frame: int) -> tuple[np.ndarray, np.ndarray, str]:
    stem = f"{frame:05d}"
    image_path = root / "images" / f"{stem}.png"
    mask_path = root / "masks" / f"{stem}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch for frame {stem}: image={image.shape[:2]}, mask={mask.shape[:2]}")
    return image, mask != 0, stem


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    tint = np.zeros_like(result)
    tint[mask] = (40, 210, 40)  # green in BGR
    result = cv2.addWeighted(result, 0.65, tint, 0.35, 0.0)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    result[boundary != 0] = (0, 0, 255)  # red contour in BGR
    return result


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(result, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def resize_panel(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = round(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []

    for frame in args.frames:
        image, mask, stem = load_pair(args.root, frame)
        overlay = overlay_mask(image, mask)
        cv2.imwrite(str(args.output / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(args.output / f"{stem}_mask.png"), (mask.astype(np.uint8) * 255))

        panels.extend(
            [
                resize_panel(add_label(image, f"original {stem}"), args.panel_width),
                resize_panel(add_label((mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=2), f"mask {stem}"), args.panel_width),
                resize_panel(add_label(overlay, f"overlay {stem}"), args.panel_width),
            ]
        )

    rows = [np.concatenate(panels[i : i + 3], axis=1) for i in range(0, len(panels), 3)]
    sheet = np.concatenate(rows, axis=0)
    sheet_path = args.output / "contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError(f"Failed to write {sheet_path}")
    print(f"Wrote {len(args.frames)} overlays and {sheet_path}")


if __name__ == "__main__":
    main()

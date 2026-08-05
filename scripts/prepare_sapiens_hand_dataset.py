#!/usr/bin/env python3
"""Build a hand-only MHR dataset from a hand-visibility split.

SapiensV2's 29-class body-part model assigns class 6 to ``Left_Hand`` and
class 15 to ``Right_Hand``.  This script keeps those pixels for frames in the
split's ``valid`` list, writes lossless PNG images with every other pixel set
to exactly zero, and symlinks the selected frame's unchanged dataset files.

Run this script with the Python environment belonging to the local SapiensV2
checkout, for example ``/path/to/scanner/.venv/bin/python``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


HAND_CLASS_IDS = (6, 15)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sapiens2-root", type=Path, required=True)
    parser.add_argument("--seg-config", type=Path, required=True)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-name", default="Camera_B1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panel-count", type=int, default=8)
    args = parser.parse_args()
    if args.panel_count < 1:
        parser.error("--panel-count must be at least 1")
    return args


def source_image(image_dir: Path, frame_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{frame_id}{suffix}"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No image for {frame_id} in {image_dir}")


def ensure_symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.is_symlink():
        if target.resolve() == source:
            return
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Conflicting symlink: {target}")
    elif target.exists():
        raise FileExistsError(f"Refusing to replace existing path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def selected_file(directory: Path, frame_id: str) -> Path | None:
    matches = [path for path in directory.glob(f"{frame_id}.*") if path.is_file()]
    if len(matches) > 1:
        raise ValueError(f"Multiple files for {frame_id} in {directory}: {matches}")
    return matches[0] if matches else None


def link_selected_directory(source: Path, target: Path, frame_ids: list[str]) -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame_id in frame_ids:
        path = selected_file(source, frame_id)
        if path is not None:
            ensure_symlink(path, target / path.name)
            count += 1
    return count


def build_symlink_skeleton(
    source_camera: Path,
    output_root: Path,
    camera_name: str,
    frame_ids: list[str],
) -> dict[str, int]:
    source_subject = source_camera.parent
    output_root.mkdir(parents=True, exist_ok=True)
    # Preserve subject-level annotations/canonical meshes and unused cameras as
    # direct symlinks.  Camera_B1 is rebuilt selectively below.
    for child in source_subject.iterdir():
        if child.name != camera_name:
            ensure_symlink(child, output_root / child.name)

    target_camera = output_root / camera_name
    target_camera.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for child in source_camera.iterdir():
        if child.name in {"images", "hand_masks", "results"}:
            continue
        if child.is_dir() and child.name in {"masks", "masks_cihp_orig_backup"}:
            counts[child.name] = link_selected_directory(
                child, target_camera / child.name, frame_ids
            )
        else:
            ensure_symlink(child, target_camera / child.name)

    source_results = source_camera / "results"
    target_results = target_camera / "results"
    target_results.mkdir(parents=True, exist_ok=True)
    for child in source_results.iterdir():
        if child.is_dir():
            counts[f"results/{child.name}"] = link_selected_directory(
                child, target_results / child.name, frame_ids
            )
        else:
            ensure_symlink(child, target_results / child.name)
    return counts


class SapiensHandSegmenter:
    def __init__(self, root: Path, config: Path, checkpoint: Path, device: str) -> None:
        sys.path.insert(0, str(root.expanduser().resolve()))
        from sapiens.dense.models import init_model

        self.device = torch.device(device)
        self.model = init_model(
            config.expanduser().resolve(), checkpoint.expanduser().resolve(), device=device
        )

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        data: Any = self.model.pipeline(dict(img=image_bgr))
        data = self.model.data_preprocessor(data)
        with torch.inference_mode():
            logits = self.model(data["inputs"])
            logits = F.interpolate(
                logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
            )
        labels = logits.argmax(dim=1)[0].cpu().numpy()
        return np.isin(labels, HAND_CLASS_IDS)


def panel_tile(image: np.ndarray, mask: np.ndarray, masked: np.ndarray, frame_id: str) -> np.ndarray:
    mask_bgr = np.zeros_like(image)
    mask_bgr[mask] = (255, 255, 255)
    panels = (image, mask_bgr, masked)
    labels = (f"{frame_id} original", "SapiensV2 hands", "absolute-black background")
    rendered = []
    for panel, label in zip(panels, labels):
        scale = 280 / panel.shape[1]
        tile = cv2.resize(panel, (280, round(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(tile)
    return np.concatenate(rendered, axis=1)


def main() -> int:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("SapiensV2 hand segmentation requires an available CUDA device")
    split_path = args.split_json.expanduser().resolve()
    split = json.loads(split_path.read_text())
    frame_ids = [str(frame_id) for frame_id in split["valid"]]
    if not frame_ids:
        raise ValueError(f"The split has no valid frames: {split_path}")
    source_camera = Path(split["input_root"]).expanduser().resolve()
    if source_camera.name != args.camera_name:
        raise ValueError(
            f"Split camera is {source_camera.name}, expected {args.camera_name}"
        )
    output_root = args.output_root.expanduser().resolve()
    link_counts = build_symlink_skeleton(
        source_camera, output_root, args.camera_name, frame_ids
    )

    segmenter = SapiensHandSegmenter(
        args.sapiens2_root, args.seg_config, args.seg_checkpoint, args.device
    )
    output_camera = output_root / args.camera_name
    image_dir = output_camera / "images"
    mask_dir = output_camera / "hand_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    preview_rows: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []

    for index, frame_id in enumerate(frame_ids, start=1):
        output_image = image_dir / f"{frame_id}.png"
        output_mask = mask_dir / f"{frame_id}.png"
        original_path = source_image(source_camera / "images", frame_id)
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(original_path)
        if output_image.exists() and output_mask.exists() and not args.overwrite:
            hand_mask = cv2.imread(str(output_mask), cv2.IMREAD_GRAYSCALE) != 0
            masked = cv2.imread(str(output_image), cv2.IMREAD_COLOR)
            if masked is None:
                raise FileNotFoundError(output_image)
        else:
            hand_mask = segmenter.predict(original)
            if not hand_mask.any():
                raise RuntimeError(f"SapiensV2 found no hand pixels in {frame_id}")
            masked = np.zeros_like(original)
            masked[hand_mask] = original[hand_mask]
            if not cv2.imwrite(str(output_mask), hand_mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Failed to write {output_mask}")
            if not cv2.imwrite(str(output_image), masked):
                raise RuntimeError(f"Failed to write {output_image}")
        outside_nonzero = int(np.count_nonzero(masked[~hand_mask]))
        if outside_nonzero != 0:
            raise RuntimeError(
                f"{output_image} has {outside_nonzero} nonzero values outside its hand mask"
            )
        fraction = float(hand_mask.mean())
        frame_stats.append(
            {"frame_id": frame_id, "hand_pixel_fraction": fraction, "outside_nonzero": 0}
        )
        if len(preview_rows) < args.panel_count:
            preview_rows.append(panel_tile(original, hand_mask, masked, frame_id))
        print(
            f"[{index:03d}/{len(frame_ids):03d}] {frame_id}: "
            f"hand pixels={hand_mask.sum()} ({fraction:.5f})",
            flush=True,
        )

    panel = np.concatenate(preview_rows, axis=0)
    panel_path = output_root / "hand_dataset_panel.png"
    if not cv2.imwrite(str(panel_path), panel):
        raise RuntimeError(f"Failed to write {panel_path}")
    manifest = {
        "source_split": str(split_path),
        "source_camera": str(source_camera),
        "camera_name": args.camera_name,
        "frame_ids": frame_ids,
        "hand_class_ids": list(HAND_CLASS_IDS),
        "hand_class_names": ["Left_Hand", "Right_Hand"],
        "symlink_counts": link_counts,
        "frames": frame_stats,
    }
    (output_root / "hand_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote hand-only dataset: {output_root}")
    print(f"Wrote validation panel: {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

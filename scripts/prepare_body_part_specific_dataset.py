#!/usr/bin/env python3
"""Build a body-part-only MHR dataset from a body-visibility split.

SapiensV2's 29-class body-part model (``DOME_CLASSES_29``) assigns pixel
classes to named body regions.  This script keeps the pixels of the chosen
``--body-part`` for frames in the split's ``valid`` list, writes lossless PNG
images with every other pixel set to exactly zero, and symlinks the selected
frame's unchanged dataset files.

Region disambiguation (e.g. left vs. right hand) happens entirely here, at
preprocessing time: alongside the combined ``<part>_masks/`` (union of every
region, used for whole-part training), this script also writes one
``<part>_masks/<region>/`` directory per region (e.g. ``left``/``right`` for a
lateral part such as "hand" or "arm") -- each containing only that region's
Sapiens classes, so single-region training at ``scripts/train.py`` just reads
the matching directory instead of disambiguating blobs at every iteration.

Run this script with the Python environment belonging to the local SapiensV2
checkout, for example ``/path/to/scanner/.venv/bin/python``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Sapiens class ids per region, from SapiensV2's DOME_CLASSES_29 (dome29
# palette). Lateral parts have "left"/"right" regions; non-lateral parts have
# a single region whose name equals the part name. "arm"/"leg" combine the
# lower + upper Sapiens classes, matching how src/body_models' left_arm/
# right_arm/left_leg/right_leg vertex groups each span more than one Sapiens
# limb segment. "head" is everything DOME_CLASSES_29 segments as part of the
# face/scalp -- Face_Neck, Hair, Eyeglass, and the mouth-interior classes
# (lips/teeth/tongue) -- since the MHR head vertex group (joint 110's
# subtree) covers the whole head and Sapiens splits it into many fine-
# grained classes rather than one.
BODY_PART_REGIONS: dict[str, dict[str, tuple[int, ...]]] = {
    "hand": {"left": (6,), "right": (15,)},
    "arm": {"left": (7, 11), "right": (16, 20)},
    "leg": {"left": (8, 12), "right": (17, 21)},
    "foot": {"left": (5,), "right": (14,)},
    "torso": {"torso": (22,)},
    "head": {"head": (2, 3, 4, 24, 25, 26, 27, 28)},
}
DOME_CLASS_NAMES = {
    2: "Eyeglass",
    3: "Face_Neck",
    4: "Hair",
    5: "Left_Foot",
    6: "Left_Hand",
    7: "Left_Lower_Arm",
    8: "Left_Lower_Leg",
    11: "Left_Upper_Arm",
    12: "Left_Upper_Leg",
    14: "Right_Foot",
    15: "Right_Hand",
    16: "Right_Lower_Arm",
    17: "Right_Lower_Leg",
    20: "Right_Upper_Arm",
    21: "Right_Upper_Leg",
    22: "Torso",
    24: "Lower_Lip",
    25: "Upper_Lip",
    26: "Lower_Teeth",
    27: "Upper_Teeth",
    28: "Tongue",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sapiens2-root", type=Path, required=True)
    parser.add_argument("--seg-config", type=Path, required=True)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--body-part",
        default="hand",
        choices=sorted(BODY_PART_REGIONS),
        help="Body part to isolate (default: hand).",
    )
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
    mask_dir_name: str,
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
        if child.name in {"images", mask_dir_name, "results"}:
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


class SapiensBodyPartSegmenter:
    def __init__(
        self,
        root: Path,
        config: Path,
        checkpoint: Path,
        device: str,
        class_ids_by_region: dict[str, tuple[int, ...]],
    ) -> None:
        sys.path.insert(0, str(root.expanduser().resolve()))
        from sapiens.dense.models import init_model

        self.class_ids_by_region = class_ids_by_region
        self.device = torch.device(device)
        self.model = init_model(
            config.expanduser().resolve(), checkpoint.expanduser().resolve(), device=device
        )

    def predict(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Per-region boolean masks, keyed by this segmenter's region names."""
        data: Any = self.model.pipeline(dict(img=image_bgr))
        data = self.model.data_preprocessor(data)
        with torch.inference_mode():
            logits = self.model(data["inputs"])
            logits = F.interpolate(
                logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
            )
        labels = logits.argmax(dim=1)[0].cpu().numpy()
        return {
            region: np.isin(labels, class_ids)
            for region, class_ids in self.class_ids_by_region.items()
        }


def panel_tile(image: np.ndarray, mask: np.ndarray, masked: np.ndarray, frame_id: str, part_label: str) -> np.ndarray:
    mask_bgr = np.zeros_like(image)
    mask_bgr[mask] = (255, 255, 255)
    panels = (image, mask_bgr, masked)
    labels = (f"{frame_id} original", f"SapiensV2 {part_label}", "absolute-black background")
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
        raise RuntimeError("SapiensV2 body-part segmentation requires an available CUDA device")
    regions = BODY_PART_REGIONS[args.body_part]
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
    mask_dir_name = f"{args.body_part}_masks"
    link_counts = build_symlink_skeleton(
        source_camera, output_root, args.camera_name, frame_ids, mask_dir_name
    )

    segmenter = SapiensBodyPartSegmenter(
        args.sapiens2_root, args.seg_config, args.seg_checkpoint, args.device, regions
    )
    output_camera = output_root / args.camera_name
    image_dir = output_camera / "images"
    mask_dir = output_camera / mask_dir_name
    region_mask_dirs = {region: mask_dir / region for region in regions}
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for region_dir in region_mask_dirs.values():
        region_dir.mkdir(parents=True, exist_ok=True)
    preview_rows: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []

    for index, frame_id in enumerate(frame_ids, start=1):
        output_image = image_dir / f"{frame_id}.png"
        output_mask = mask_dir / f"{frame_id}.png"
        region_outputs = {region: d / f"{frame_id}.png" for region, d in region_mask_dirs.items()}
        original_path = source_image(source_camera / "images", frame_id)
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(original_path)
        existing = output_image.exists() and output_mask.exists() and all(
            p.exists() for p in region_outputs.values()
        )
        if existing and not args.overwrite:
            part_mask = cv2.imread(str(output_mask), cv2.IMREAD_GRAYSCALE) != 0
            region_masks = {
                region: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) != 0
                for region, p in region_outputs.items()
            }
            masked = cv2.imread(str(output_image), cv2.IMREAD_COLOR)
            if masked is None:
                raise FileNotFoundError(output_image)
        else:
            region_masks = segmenter.predict(original)
            part_mask = np.logical_or.reduce(list(region_masks.values()))
            if not part_mask.any():
                raise RuntimeError(f"SapiensV2 found no {args.body_part} pixels in {frame_id}")
            masked = np.zeros_like(original)
            masked[part_mask] = original[part_mask]
            if not cv2.imwrite(str(output_mask), part_mask.astype(np.uint8) * 255):
                raise RuntimeError(f"Failed to write {output_mask}")
            for region, region_mask in region_masks.items():
                if not cv2.imwrite(str(region_outputs[region]), region_mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"Failed to write {region_outputs[region]}")
            if not cv2.imwrite(str(output_image), masked):
                raise RuntimeError(f"Failed to write {output_image}")
        outside_nonzero = int(np.count_nonzero(masked[~part_mask]))
        if outside_nonzero != 0:
            raise RuntimeError(
                f"{output_image} has {outside_nonzero} nonzero values outside its {args.body_part} mask"
            )
        region_names = list(region_masks)
        for i, region_a in enumerate(region_names):
            for region_b in region_names[i + 1 :]:
                overlap = np.logical_and(region_masks[region_a], region_masks[region_b])
                if overlap.any():
                    raise RuntimeError(
                        f"{frame_id}: {region_a}/{region_b} masks overlap on "
                        f"{int(overlap.sum())} pixels"
                    )
        fraction = float(part_mask.mean())
        stats = {
            "frame_id": frame_id,
            f"{args.body_part}_pixel_fraction": fraction,
            "outside_nonzero": 0,
        }
        for region, region_mask in region_masks.items():
            stats[f"{region}_pixel_fraction"] = float(region_mask.mean())
        frame_stats.append(stats)
        if len(preview_rows) < args.panel_count:
            preview_rows.append(panel_tile(original, part_mask, masked, frame_id, args.body_part))
        region_summary = ", ".join(
            f"{region}={region_masks[region].sum()}" for region in region_names
        )
        print(
            f"[{index:03d}/{len(frame_ids):03d}] {frame_id}: "
            f"{args.body_part} pixels={part_mask.sum()} ({fraction:.5f}), {region_summary}",
            flush=True,
        )

    panel = np.concatenate(preview_rows, axis=0)
    panel_path = output_root / f"{args.body_part}_dataset_panel.png"
    if not cv2.imwrite(str(panel_path), panel):
        raise RuntimeError(f"Failed to write {panel_path}")
    manifest = {
        "source_split": str(split_path),
        "source_camera": str(source_camera),
        "camera_name": args.camera_name,
        "body_part": args.body_part,
        "frame_ids": frame_ids,
        "class_ids_by_region": {region: list(ids) for region, ids in regions.items()},
        "class_names_by_region": {
            region: [DOME_CLASS_NAMES[i] for i in ids] for region, ids in regions.items()
        },
        "mask_dirs_by_region": {region: f"{mask_dir_name}/{region}" for region in regions},
        "symlink_counts": link_counts,
        "frames": frame_stats,
    }
    (output_root / f"{args.body_part}_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote {args.body_part}-only dataset: {output_root}")
    print(f"Wrote validation panel: {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

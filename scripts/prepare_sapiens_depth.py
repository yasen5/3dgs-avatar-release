#!/usr/bin/env python3
"""CUDA-only Sapiens depth pass for masks produced by ``prepare_sapiens_masks.py``.

The script deliberately has no Sapiens2 or Transformers dependency.  Run it
from the environment compatible with the selected depth checkpoint.  For the
NeuMan 1B BF16 checkpoint used here, that is the isolated PyTorch 2.3 CUDA
environment::

    /mnt/ssd2/sapiens_bf16_venv/bin/python scripts/prepare_sapiens_depth.py \
        --output-root /mnt/ssd2/exavatar_neuman_combined/data/citron \
        --depth-checkpoint /mnt/ssd2/sapiens_checkpoints/depth/sapiens_1b_render_people_epoch_88_bfloat16.pt2

The mask pass writes one crop metadata JSON per frame.  Reading that metadata
means this pass does not need to import the dataset, SMPL-X, Sapiens2, or the
segmentation environment.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapiens_ga_avatar_common import read_crop_metadata

LOGGER = logging.getLogger("prepare_sapiens_depth")
DEFAULT_DEPTH_FILE = "sapiens_1b_render_people_epoch_88_bfloat16.pt2"


def _find_image(image_root: Path, frame_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_root / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no image found for frame {frame_id} in {image_root}")


def _numeric_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 2**63 - 1, path.stem


def _depth_context(image: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a full-frame depth visualization over the original image.

    The depth array is deliberately not cropped here: its shape is the source
    image shape, and only foreground pixels are colorized.  Keeping the
    background as the original image makes crop placement and camera context
    directly visible during validation.
    """
    valid = (mask > 0) & np.isfinite(depth)
    color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        lo, hi = np.percentile(values, [2.0, 98.0]).astype(np.float32)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        scale = max(float(hi - lo), 1e-6)
        normalized = np.clip((depth - lo) / scale, 0.0, 1.0)
        color = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    context = image.copy()
    blended = cv2.addWeighted(image, 0.35, color, 0.65, 0.0)
    context[valid] = blended[valid]
    return context


def _resolve_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"depth checkpoint not found: {path}")
        return path
    env_path = os.environ.get("SAPIENS_DEPTH_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()
    raise FileNotFoundError(
        f"pass --depth-checkpoint or set SAPIENS_DEPTH_CHECKPOINT; default file is {DEFAULT_DEPTH_FILE}"
    )


class SapiensDepth:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(device)
        LOGGER.info("Loading Sapiens depth checkpoint %s", checkpoint)
        with zipfile.ZipFile(checkpoint) as archive:
            is_exported_program = "serialized_exported_program.json" in archive.namelist()
        try:
            if is_exported_program:
                # The BF16 archive is exported with PyTorch 2.3.  This script
                # is intentionally launched by that compatible environment.
                self.model = torch.export.load(str(checkpoint)).module().to(self.device)
                self.input_dtype = torch.bfloat16
            else:
                self.model = torch.jit.load(str(checkpoint), map_location=self.device).eval().to(self.device)
                self.input_dtype = torch.float32
        except Exception as exc:
            if is_exported_program:
                raise RuntimeError(
                    "Could not load the exported Sapiens depth checkpoint. "
                    "Run prepare_sapiens_depth.py with the compatible PyTorch 2.3 CUDA environment."
                ) from exc
            raise

    def predict(self, masked_crop_bgr: np.ndarray) -> np.ndarray:
        height, width = masked_crop_bgr.shape[:2]
        image = cv2.resize(masked_crop_bgr, (768, 1024), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy())
        image = image.to(device=self.device, dtype=torch.float32)
        mean = torch.tensor([123.5, 116.5, 103.5], device=self.device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([58.5, 57.0, 57.5], device=self.device, dtype=torch.float32).view(3, 1, 1)
        image = ((image - mean) / std).unsqueeze(0).to(dtype=self.input_dtype)
        with torch.inference_mode():
            output = self.model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim == 3:
                output = output.unsqueeze(1)
            output = F.interpolate(
                output.float(), size=(height, width), mode="bilinear", align_corners=False
            )
        return output[0, 0].detach().cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="directory containing frame images; defaults to output-root/images")
    parser.add_argument("--mask-dir", type=Path, help="defaults to output-root/masks")
    parser.add_argument("--metadata-dir", type=Path, help="defaults to output-root/sapiens_ga_avatar_crops")
    parser.add_argument("--depth-dir", type=Path, help="defaults to output-root/sapiens_depth")
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda", help="CUDA device for Sapiens depth")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Sapiens depth generation requires CUDA; CPU inference is unsupported")
    if args.start < 0 or args.stride < 1:
        raise ValueError("start must be non-negative and stride must be positive")

    output_root = args.output_root.expanduser().resolve()
    image_root = (args.image_root or output_root / "images").expanduser().resolve()
    mask_dir = (args.mask_dir or output_root / "masks").expanduser().resolve()
    metadata_dir = (args.metadata_dir or output_root / "sapiens_ga_avatar_crops").expanduser().resolve()
    depth_dir = (args.depth_dir or output_root / "sapiens_depth").expanduser().resolve()
    metadata_paths = sorted(metadata_dir.glob("*.json"), key=_numeric_frame_key)
    metadata_paths = metadata_paths[args.start : args.end : args.stride]
    if not metadata_paths:
        raise RuntimeError(
            f"no crop metadata found in {metadata_dir}; run prepare_sapiens_masks.py first"
        )

    checkpoint = _resolve_checkpoint(args.depth_checkpoint)
    depth_model = SapiensDepth(checkpoint, args.device)
    LOGGER.info("Processing %d frame(s)", len(metadata_paths))

    for index, metadata_path in enumerate(metadata_paths, start=1):
        frame_id = metadata_path.stem
        mask_path = mask_dir / f"{frame_id}.png"
        depth_path = depth_dir / f"{frame_id}.npy"
        if depth_path.exists() and not args.overwrite:
            if args.overlay_dir is not None:
                image_path = _find_image(image_root, frame_id)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                depth = np.load(depth_path).astype(np.float32)
                if image is None or mask is None:
                    raise RuntimeError(f"could not read image/mask for existing depth map {frame_id}")
                if image.shape[:2] != mask.shape or depth.shape != image.shape[:2]:
                    raise ValueError(
                        f"existing depth context shape mismatch for {frame_id}: "
                        f"image={image.shape[:2]}, mask={mask.shape}, depth={depth.shape}"
                    )
                overlay = _depth_context(image, depth, mask)
                args.overlay_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.overlay_dir / f"{frame_id}.jpg"), overlay):
                    raise RuntimeError(f"failed to write depth context: {args.overlay_dir / f'{frame_id}.jpg'}")
            LOGGER.info("[%d/%d] skip %s", index, len(metadata_paths), frame_id)
            continue

        image_path = _find_image(image_root, frame_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(
                f"mask not found: {mask_path}; run prepare_sapiens_masks.py first"
            )
        if mask.shape != image.shape[:2]:
            raise ValueError(f"mask/image shape mismatch for {frame_id}: {mask.shape} vs {image.shape[:2]}")
        x0, y0, x1, y1 = read_crop_metadata(metadata_path, image.shape[:2])
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1] > 0
        if not np.any(crop_mask):
            raise RuntimeError(f"mask has no foreground pixels in frame {frame_id}")
        masked_crop = np.zeros_like(crop)
        masked_crop[crop_mask] = crop[crop_mask]
        crop_depth = depth_model.predict(masked_crop)

        full_depth = np.zeros(image.shape[:2], dtype=np.float32)
        full_depth[y0:y1, x0:x1] = np.where(crop_mask, crop_depth, 0.0)
        depth_dir.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, full_depth)

        if args.overlay_dir is not None:
            overlay = _depth_context(image, full_depth, mask)
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.overlay_dir / f"{frame_id}.jpg"), overlay):
                raise RuntimeError(f"failed to write depth context: {args.overlay_dir / f'{frame_id}.jpg'}")
        LOGGER.info("[%d/%d] %s crop=(%d,%d)-(%d,%d)", index, len(metadata_paths), frame_id, x0, y0, x1, y1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

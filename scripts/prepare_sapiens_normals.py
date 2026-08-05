#!/usr/bin/env python3
"""CUDA-only Sapiens surface-normal pass for GA-Avatar.

The normal model is run on the same masked SMPL crop used by the depth pass.
Its prediction is then put back into the original image canvas, so every
``sapiens_normal/<frame>.png`` is full resolution rather than crop resolution.
The background is neutral encoded normal ``[128, 128, 128]`` and is ignored by
GA-Avatar's foreground loss through the NeuMan mask.

Example::

    /mnt/ssd2/sapiens_bf16_venv/bin/python scripts/prepare_sapiens_normals.py \
        --output-root /mnt/ssd2/exavatar_neuman_combined/data/citron \
        --normal-checkpoint /mnt/ssd2/sapiens_checkpoints/normal/sapiens_0.3b_normal_torchscript.pt2

The checkpoint is the locally available Sapiens 0.3B TorchScript normal model.
Inference is CUDA-only; this script refuses to fall back to CPU.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapiens_ga_avatar_common import read_crop_metadata

LOGGER = logging.getLogger("prepare_sapiens_normals")


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


def _resolve_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"normal checkpoint not found: {path}")
        return path
    env_path = os.environ.get("SAPIENS_NORMAL_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()
    raise FileNotFoundError(
        "pass --normal-checkpoint or set SAPIENS_NORMAL_CHECKPOINT"
    )


class SapiensNormals:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(device)
        LOGGER.info("Loading Sapiens normal checkpoint %s", checkpoint)
        self.model = torch.jit.load(str(checkpoint), map_location=self.device).eval().to(self.device)
        self.mean = torch.tensor(
            [123.5, 116.5, 103.5], device=self.device, dtype=torch.float32
        ).view(3, 1, 1)
        self.std = torch.tensor(
            [58.5, 57.0, 57.5], device=self.device, dtype=torch.float32
        ).view(3, 1, 1)

    def predict(self, masked_crop_bgr: np.ndarray) -> np.ndarray:
        height, width = masked_crop_bgr.shape[:2]
        resized = cv2.resize(masked_crop_bgr, (768, 1024), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(resized[..., ::-1])
        image = torch.from_numpy(rgb.transpose(2, 0, 1)).to(
            device=self.device, dtype=torch.float32
        )
        image = ((image - self.mean) / self.std).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 4 or output.shape[1] != 3:
                raise RuntimeError(f"unexpected Sapiens normal output shape: {tuple(output.shape)}")
            output = F.interpolate(
                output.float(), size=(height, width), mode="bilinear", align_corners=False
            )[0]
            output = F.normalize(output, dim=0, eps=1e-6)
        return output.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _normal_context(image: np.ndarray, normal_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    encoded_bgr = np.clip((normal_rgb + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)[..., ::-1]
    valid = mask > 0
    context = image.copy()
    blended = cv2.addWeighted(image, 0.35, encoded_bgr, 0.65, 0.0)
    context[valid] = blended[valid]
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="directory containing frame images; defaults to output-root/images")
    parser.add_argument("--mask-dir", type=Path, help="defaults to output-root/masks")
    parser.add_argument("--metadata-dir", type=Path, help="defaults to output-root/sapiens_ga_avatar_crops")
    parser.add_argument("--normal-dir", type=Path, help="defaults to output-root/sapiens_normal")
    parser.add_argument("--normal-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda", help="CUDA device for Sapiens normals")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, help="full-frame normal-over-image validation output")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Sapiens normal generation requires CUDA; CPU inference is unsupported")
    if args.start < 0 or args.stride < 1:
        raise ValueError("start must be non-negative and stride must be positive")

    output_root = args.output_root.expanduser().resolve()
    image_root = (args.image_root or output_root / "images").expanduser().resolve()
    mask_dir = (args.mask_dir or output_root / "masks").expanduser().resolve()
    metadata_dir = (args.metadata_dir or output_root / "sapiens_ga_avatar_crops").expanduser().resolve()
    normal_dir = (args.normal_dir or output_root / "sapiens_normal").expanduser().resolve()
    metadata_paths = sorted(metadata_dir.glob("*.json"), key=_numeric_frame_key)
    metadata_paths = metadata_paths[args.start : args.end : args.stride]
    if not metadata_paths:
        raise RuntimeError(f"no crop metadata found in {metadata_dir}; run prepare_sapiens_masks.py first")

    checkpoint = _resolve_checkpoint(args.normal_checkpoint)
    normal_model = SapiensNormals(checkpoint, args.device)
    LOGGER.info("Processing %d frame(s)", len(metadata_paths))

    for index, metadata_path in enumerate(metadata_paths, start=1):
        frame_id = metadata_path.stem
        normal_path = normal_dir / f"{frame_id}.png"
        image_path = _find_image(image_root, frame_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_path = mask_dir / f"{frame_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"mask not found: {mask_path}; run prepare_sapiens_masks.py first")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"mask/image shape mismatch for {frame_id}: {mask.shape} vs {image.shape[:2]}")

        if normal_path.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] skip %s", index, len(metadata_paths), frame_id)
            if args.overlay_dir is None:
                continue
            encoded = cv2.imread(str(normal_path), cv2.IMREAD_COLOR)
            if encoded is None:
                raise RuntimeError(f"could not read existing normal map: {normal_path}")
            overlay = image.copy()
            valid = mask > 0
            overlay[valid] = cv2.addWeighted(image, 0.35, encoded, 0.65, 0.0)[valid]
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.overlay_dir / f"{frame_id}.jpg"), overlay)
            continue

        x0, y0, x1, y1 = read_crop_metadata(metadata_path, image.shape[:2])
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1] > 0
        if not np.any(crop_mask):
            raise RuntimeError(f"mask has no foreground pixels in frame {frame_id}")
        masked_crop = np.zeros_like(crop)
        masked_crop[crop_mask] = crop[crop_mask]
        crop_normal = normal_model.predict(masked_crop)

        full_normal = np.zeros((*image.shape[:2], 3), dtype=np.float32)
        full_normal[y0:y1, x0:x1] = np.where(crop_mask[..., None], crop_normal, 0.0)
        encoded_rgb = np.clip((full_normal + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        encoded_bgr = encoded_rgb[..., ::-1]
        normal_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(normal_path), encoded_bgr):
            raise RuntimeError(f"failed to write normal map: {normal_path}")

        if args.overlay_dir is not None:
            overlay = _normal_context(image, full_normal, mask)
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.overlay_dir / f"{frame_id}.jpg"), overlay):
                raise RuntimeError(f"failed to write normal context: {args.overlay_dir / f'{frame_id}.jpg'}")
        LOGGER.info("[%d/%d] %s full_canvas=%dx%d crop=(%d,%d)-(%d,%d)", index, len(metadata_paths), frame_id, image.shape[1], image.shape[0], x0, y0, x1, y1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

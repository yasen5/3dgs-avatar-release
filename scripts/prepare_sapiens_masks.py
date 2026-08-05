#!/usr/bin/env python3
"""CUDA-only Sapiens2 segmentation pass for GA-Avatar.

This pass writes the full-resolution foreground masks and the crop metadata
consumed by :mod:`prepare_sapiens_depth`.  It does not import or load a depth
model, so it can run in the normal Sapiens2 environment independently.

NeuMan/SMPL-X example::

    python scripts/prepare_sapiens_masks.py \
        --backend smplx \
        --dataset-config configs/dataset/neuman_citron_smplx.yaml \
        --dataset-root /mnt/ssd2/exavatar_neuman_combined/data/citron \
        --output-root /mnt/ssd2/exavatar_neuman_combined/data/citron \
        --sapiens2-root /home/yasen/VesalientSkinBodyScanner/third_party/sapiens2 \
        --seg-checkpoint /home/yasen/VesalientSkinBodyScanner/models/sapiens2/0.4b/model.safetensors
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sapiens_ga_avatar_common import (
    build_source,
    frame_crop,
    selected_frames,
    write_crop_metadata,
)

LOGGER = logging.getLogger("prepare_sapiens_masks")
DEFAULT_SEG_MODEL = "facebook/sapiens2-seg-0.4b"


def _find_sapiens2_root(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            Path.home() / "sapiens2",
            Path.home() / "VesalientSkinBodyScanner" / "third_party" / "sapiens2",
        ]
    )
    for candidate in candidates:
        if (candidate / "sapiens" / "dense").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the local Sapiens2 checkout; pass --sapiens2-root."
    )


def _find_sapiens2_files(root: Path, model_id: str, checkpoint: Path | None) -> tuple[Path, Path]:
    size = model_id.rsplit("-", 1)[-1]
    config = root / "sapiens" / "dense" / "configs" / "seg" / "shutterstock_goliath" / (
        f"sapiens2_{size}_seg_shutterstock_goliath-1024x768.py"
    )
    if checkpoint is None:
        candidates = [
            root.parent.parent / "models" / "sapiens2" / size / "model.safetensors",
            root / "models" / "sapiens2" / size / "model.safetensors",
        ]
        checkpoint = next((path for path in candidates if path.is_file()), None)
    if not config.is_file():
        raise FileNotFoundError(f"Sapiens2 segmentation config not found: {config}")
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "Sapiens2 segmentation checkpoint is not available locally; pass --seg-checkpoint"
        )
    return config, checkpoint.resolve()


class Sapiens2Segmenter:
    def __init__(
        self,
        model_id: str,
        device: str,
        sapiens2_root: Path | None,
        checkpoint: Path | None,
    ) -> None:
        root = _find_sapiens2_root(sapiens2_root)
        config, checkpoint = _find_sapiens2_files(root, model_id, checkpoint)
        LOGGER.info("Loading Sapiens2 segmentation model %s from %s", model_id, checkpoint)
        self.processor = None
        self.backend = "official"
        hf_source = checkpoint.parent if (checkpoint.parent / "config.json").is_file() else None
        official_error: Exception | None = None
        try:
            if hf_source is None:
                raise ImportError("no local Transformers config beside checkpoint")
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            self.processor = AutoImageProcessor.from_pretrained(
                str(hf_source), local_files_only=True
            )
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                str(hf_source), local_files_only=True
            )
            self.backend = "transformers"
        except Exception as exc:
            official_error = exc
        if self.backend == "official":
            sys.path.insert(0, str(root))
            try:
                from sapiens.dense.models import init_model

                self.model = init_model(config, checkpoint, device=device)
            except Exception as exc:
                raise RuntimeError(
                    "The local Sapiens2 implementation could not be loaded. "
                    "Use the compatible Sapiens2 environment."
                ) from (official_error or exc)
        self.device = torch.device(device)
        self.model.eval().to(self.device)

    def predict(self, crop_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "transformers":
            rgb = np.ascontiguousarray(crop_bgr[..., ::-1])
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        else:
            data: Any = self.model.pipeline(dict(img=crop_bgr))
            data = self.model.data_preprocessor(data)
        with torch.inference_mode():
            if self.backend == "transformers":
                logits = self.model(**inputs).logits
            else:
                logits = self.model(data["inputs"])
            logits = F.interpolate(
                logits, size=crop_bgr.shape[:2], mode="bilinear", align_corners=False
            )
        return logits.argmax(dim=1)[0].cpu().numpy() != 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("mhr", "smplx"), required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "predict"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--device", default="cuda", help="CUDA device for Sapiens2 segmentation")
    parser.add_argument("--seg-model", default=DEFAULT_SEG_MODEL)
    parser.add_argument("--sapiens2-root", type=Path)
    parser.add_argument("--seg-checkpoint", type=Path)
    parser.add_argument("--mhr-model", type=Path)
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
        raise RuntimeError("Sapiens2 mask generation requires CUDA; CPU inference is unsupported")
    if args.margin < 0:
        raise ValueError("margin must be non-negative")

    source = build_source(
        args.backend,
        args.input_root,
        args.dataset_config,
        args.dataset_root,
        args.split,
        args.mhr_model,
    )
    frames = selected_frames(source, args.start, args.end, args.stride)
    segmenter = Sapiens2Segmenter(args.seg_model, args.device, args.sapiens2_root, args.seg_checkpoint)
    mask_dir = args.output_root.expanduser().resolve() / "masks"
    metadata_dir = args.output_root.expanduser().resolve() / "sapiens_ga_avatar_crops"
    LOGGER.info("Processing %d frame(s)", len(frames))

    for index, frame in enumerate(frames, start=1):
        mask_path = mask_dir / f"{frame.frame_id}.png"
        metadata_path = metadata_dir / f"{frame.frame_id}.json"
        if mask_path.exists() and metadata_path.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] skip %s", index, len(frames), frame.frame_id)
            continue
        image, box = frame_crop(frame, args.margin)
        x0, y0, x1, y1 = box
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = segmenter.predict(crop)
        if not np.any(crop_mask):
            raise RuntimeError(f"Sapiens2 found no foreground pixels in frame {frame.frame_id}")

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = crop_mask.astype(np.uint8) * 255
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), full_mask):
            raise RuntimeError(f"failed to write mask: {mask_path}")
        write_crop_metadata(metadata_path, frame, image.shape[:2], box)

        if args.overlay_dir is not None:
            overlay = image.copy()
            overlay[full_mask == 0] = (overlay[full_mask == 0] * 0.25).astype(np.uint8)
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.overlay_dir / f"{frame.frame_id}.jpg"), overlay)
        LOGGER.info(
            "[%d/%d] %s crop=(%d,%d)-(%d,%d) mask_area=%.4f",
            index,
            len(frames),
            frame.frame_id,
            *box,
            float(crop_mask.mean()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

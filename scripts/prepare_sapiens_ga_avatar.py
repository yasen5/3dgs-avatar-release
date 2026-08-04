#!/usr/bin/env python3
"""Prepare GA-Avatar masks and Sapiens depth maps from posed body meshes.

The script deliberately runs the estimators on a mesh-derived person crop,
then restores both predictions to the original frame size expected by
GA-Avatar.  Sapiens2 segmentation is loaded from the local Hugging Face cache
only.  The depth estimator is the original Sapiens depth model in its official
TorchScript/Lite format; a missing depth checkpoint is downloaded once through
the Hugging Face cache unless ``--no-download-depth`` is supplied.

MHR example::

    python scripts/prepare_sapiens_ga_avatar.py \
        --backend mhr \
        --input-root /path/to/subject/Camera_B1 \
        --output-root /path/to/subject/Camera_B1

SMPL-X/NeuMan example::

    python scripts/prepare_sapiens_ga_avatar.py \
        --backend smplx \
        --dataset-config configs/dataset/neuman_citron_smplx.yaml \
        --output-root /path/to/neuman/citron
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("prepare_sapiens_ga_avatar")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SEG_MODEL = "facebook/sapiens2-seg-0.4b"
DEFAULT_DEPTH_REPO = "facebook/sapiens-depth-2b-torchscript"
DEFAULT_DEPTH_FILE = "sapiens_2b_render_people_epoch_25_torchscript.pt2"


@dataclass(frozen=True)
class FrameGeometry:
    frame_id: str
    image_path: Path
    vertices_camera: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


class FrameSource(Protocol):
    def frames(self) -> Iterator[FrameGeometry]: ...


def project_vertices(
    vertices_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Project camera-space vertices to pixel coordinates.

    Invalid and behind-camera vertices are returned as NaN, so callers cannot
    accidentally include them in a crop.
    """
    vertices = np.asarray(vertices_camera, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected vertices with shape (V,3), got {vertices.shape}")
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-6)
    uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = fx * vertices[valid, 0] / vertices[valid, 2] + cx
    uv[valid, 1] = fy * vertices[valid, 1] / vertices[valid, 2] + cy
    return uv


def expanded_crop_box(
    projected_vertices: np.ndarray,
    image_width: int,
    image_height: int,
    margin: float = 0.25,
) -> tuple[int, int, int, int]:
    """Return an image-clipped ``(left, top, right, bottom)`` crop box."""
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    uv = np.asarray(projected_vertices, dtype=np.float32)
    valid = np.isfinite(uv).all(axis=1)
    if not np.any(valid):
        raise ValueError("the mesh has no finite projected vertices in front of the camera")
    points = uv[valid]
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    width = max(float(right - left), 1.0)
    height = max(float(bottom - top), 1.0)
    left -= margin * width
    right += margin * width
    top -= margin * height
    bottom += margin * height
    x0 = max(0, min(image_width - 1, int(np.floor(left))))
    y0 = max(0, min(image_height - 1, int(np.floor(top))))
    x1 = max(x0 + 1, min(image_width, int(np.ceil(right)) + 1))
    y1 = max(y0 + 1, min(image_height, int(np.ceil(bottom)) + 1))
    return x0, y0, x1, y1


def _frame_image(image_dir: Path, frame_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"no image found for frame {frame_id} in {image_dir}")


def _numeric_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 2**63 - 1, path.stem


class MHRFrameSource:
    """Read native MHR raw fits, using saved posed vertices when available."""

    def __init__(self, input_root: Path, mhr_model: Path | None = None) -> None:
        self.root = input_root.expanduser().resolve()
        self.image_dir = self.root / "images"
        self.raw_dir = self.root / "results" / "raw"
        self.mhr_model_path = mhr_model
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"MHR raw-fit directory not found: {self.raw_dir}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"MHR image directory not found: {self.image_dir}")

    def frames(self) -> Iterator[FrameGeometry]:
        for raw_path in sorted(self.raw_dir.glob("*.npz"), key=_numeric_frame_key):
            frame_id = raw_path.stem
            with np.load(raw_path, allow_pickle=False) as raw:
                if "vertices" in raw.files:
                    vertices = np.asarray(raw["vertices"], dtype=np.float32)
                else:
                    vertices = self._recompute_vertices(raw, raw_path)
                translation_name = "pred_cam_t" if "pred_cam_t" in raw.files else "cam_t"
                if translation_name not in raw.files:
                    raise KeyError(f"{raw_path} has no pred_cam_t/cam_t")
                vertices = vertices + np.asarray(raw[translation_name], dtype=np.float32).reshape(1, 3)
                focal = float(np.asarray(raw["focal_length"]).reshape(()))
                width = int(np.asarray(raw["width"]).reshape(()))
                height = int(np.asarray(raw["height"]).reshape(()))
            image_path = _frame_image(self.image_dir, frame_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(
                    f"image/raw dimensions disagree for {frame_id}: "
                    f"image={None if image is None else image.shape[:2]}, raw={(height, width)}"
                )
            yield FrameGeometry(frame_id, image_path, vertices, focal, focal, width / 2.0, height / 2.0)

    def _recompute_vertices(self, raw: Any, raw_path: Path) -> np.ndarray:
        if self.mhr_model_path is None:
            raise ValueError(
                f"{raw_path} has no saved vertices; pass --mhr-model to recompute them"
            )
        from src.body_models import mhr_lbs

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = mhr_lbs.load_mhr(str(self.mhr_model_path), device=device)
        shape = torch.from_numpy(np.asarray(raw["shape_params"], dtype=np.float32)).unsqueeze(0)
        pose = torch.from_numpy(np.asarray(raw["mhr_model_params"], dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            output = mhr_lbs.mhr_query(model, shape.to(device), pose.to(device), device=device)
        return output["verts"][0].detach().cpu().numpy().astype(np.float32)


class SMPLXFrameSource:
    """Read the repository's NeuMan SMPL-X dataset through its BodyModel."""

    def __init__(self, dataset_config: Path, split: str) -> None:
        from omegaconf import OmegaConf
        from src.dataset.neuman_smplx import NeumanSMPLXDataset

        config = OmegaConf.load(str(dataset_config))
        cfg = config.dataset
        cfg.preload = False
        cfg.data_device = "cpu"
        self.dataset = NeumanSMPLXDataset(cfg, split=split)
        self.body_model = self.dataset.body_model

    def frames(self) -> Iterator[FrameGeometry]:
        from src.body_models.base import skin_vertices

        with torch.inference_mode():
            rest_device = next(self.body_model.parameters()).device
            for frame_index, frame_id in enumerate(self.dataset.frame_ids):
                raw = self.dataset.load_raw(frame_index)
                _, bone_world = self.dataset._world_bone_transforms(frame_id)
                vertices_world = skin_vertices(self.body_model, bone_world.to(rest_device))[0]
                vertices_world = vertices_world.detach().cpu().numpy().astype(np.float32)
                vertices_camera = vertices_world @ np.asarray(raw["R"], dtype=np.float32) + np.asarray(raw["T"], dtype=np.float32)
                cam = self.dataset._colmap_cam
                image_path = Path(self.dataset._image_dir) / f"{frame_id}.png"
                yield FrameGeometry(
                    frame_id,
                    image_path,
                    vertices_camera,
                    float(cam["fx"]),
                    float(cam["fy"]),
                    float(cam["cx"]),
                    float(cam["cy"]),
                )


def _find_sapiens2_root(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if os.environ.get("SAPIENS2_ROOT"):
        candidates.append(Path(os.environ["SAPIENS2_ROOT"]))
    candidates.extend((
        Path.home() / "sapiens2",
        Path.home() / "VesalientSkinBodyScanner" / "third_party" / "sapiens2",
    ))
    for candidate in candidates:
        if (candidate / "sapiens" / "dense").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the local Sapiens2 checkout. Pass --sapiens2-root or set SAPIENS2_ROOT."
    )


def _find_sapiens2_files(root: Path, model_id: str, checkpoint: Path | None) -> tuple[Path, Path]:
    size = model_id.rsplit("-", 1)[-1]
    config = root / "sapiens" / "dense" / "configs" / "seg" / "shutterstock_goliath" / (
        f"sapiens2_{size}_seg_shutterstock_goliath-1024x768.py"
    )
    if checkpoint is None:
        checkpoint_root = Path(os.environ.get("SAPIENS_CHECKPOINT_ROOT", ""))
        candidates = [
            root.parent.parent / "models" / "sapiens2" / size / "model.safetensors",
            root / "models" / "sapiens2" / size / "model.safetensors",
            checkpoint_root / "seg" / f"sapiens2_{size}_seg.safetensors",
        ]
        checkpoint = next((path for path in candidates if path.is_file()), None)
    if not config.is_file():
        raise FileNotFoundError(f"Sapiens2 segmentation config not found: {config}")
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "Sapiens2 segmentation checkpoint is not available locally; "
            "pass --seg-checkpoint or place it under the local Sapiens2 checkpoint root"
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
        LOGGER.info("Loading local Sapiens2 segmentation model %s from %s", model_id, checkpoint)
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
                    "The local Sapiens2 implementation could not be imported or loaded. "
                    "Use a compatible Sapiens2 environment, or install Transformers >= 5 "
                    "and provide the exported local model directory."
                ) from (official_error or exc)
        if self.backend == "transformers":
            self.device = torch.device(device)
            self.model.eval().to(self.device)
        else:
            self.device = torch.device(device)
            self.model.eval().to(self.device)

    def predict(self, crop_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "transformers":
            rgb = np.ascontiguousarray(crop_bgr[..., ::-1])
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        else:
            data = self.model.pipeline(dict(img=crop_bgr))
            data = self.model.data_preprocessor(data)
        with torch.inference_mode():
            if self.backend == "transformers":
                logits = self.model(**inputs).logits
            else:
                logits = self.model(data["inputs"])
            logits = F.interpolate(logits, size=crop_bgr.shape[:2], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy() != 0


def _resolve_depth_checkpoint(explicit: Path | None, allow_download: bool) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"depth checkpoint not found: {path}")
        return path
    env_path = os.environ.get("SAPIENS_DEPTH_CHECKPOINT")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cached = sorted(cache_root.glob(
        "models--facebook--sapiens-depth-2b-torchscript/snapshots/*/" + DEFAULT_DEPTH_FILE
    ))
    if cached:
        return cached[-1].resolve()
    if not allow_download:
        raise FileNotFoundError(
            "Sapiens depth checkpoint is not cached; pass --depth-checkpoint or allow downloading"
        )
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=DEFAULT_DEPTH_REPO, filename=DEFAULT_DEPTH_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"could not download {DEFAULT_DEPTH_REPO}/{DEFAULT_DEPTH_FILE}; "
            "pass --depth-checkpoint to a local Sapiens-Lite TorchScript file"
        ) from exc
    return Path(downloaded).resolve()


class SapiensDepth:
    def __init__(self, checkpoint: Path, device: str) -> None:
        LOGGER.info("Loading Sapiens depth checkpoint %s", checkpoint)
        self.device = torch.device(device)
        self.model = torch.jit.load(str(checkpoint), map_location=self.device).eval().to(self.device)

    def predict(self, masked_crop_bgr: np.ndarray) -> np.ndarray:
        height, width = masked_crop_bgr.shape[:2]
        image = cv2.resize(masked_crop_bgr, (768, 1024), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image[..., ::-1].transpose(2, 0, 1).copy()).float()
        mean = torch.tensor([123.5, 116.5, 103.5]).view(3, 1, 1)
        std = torch.tensor([58.5, 57.0, 57.5]).view(3, 1, 1)
        image = ((image - mean) / std).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim == 3:
                output = output.unsqueeze(1)
            output = F.interpolate(output.float(), size=(height, width), mode="bilinear", align_corners=False)
        return output[0, 0].cpu().numpy().astype(np.float32)


def process_frame(
    frame: FrameGeometry,
    segmenter: Sapiens2Segmenter,
    depth_model: SapiensDepth,
    mask_path: Path,
    depth_path: Path,
    margin: float,
    overlay_path: Path | None = None,
) -> tuple[tuple[int, int, int, int], float]:
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {frame.image_path}")
    height, width = image.shape[:2]
    projected = project_vertices(frame.vertices_camera, frame.fx, frame.fy, frame.cx, frame.cy)
    box = expanded_crop_box(projected, width, height, margin)
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = segmenter.predict(crop)
    if not np.any(crop_mask):
        raise RuntimeError(f"Sapiens2 found no foreground pixels in frame {frame.frame_id}")
    masked_crop = np.zeros_like(crop)
    masked_crop[crop_mask] = crop[crop_mask]
    crop_depth = depth_model.predict(masked_crop)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = crop_mask.astype(np.uint8) * 255
    full_depth = np.zeros((height, width), dtype=np.float32)
    full_depth[y0:y1, x0:x1] = np.where(crop_mask, crop_depth, 0.0)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), full_mask):
        raise RuntimeError(f"failed to write mask: {mask_path}")
    np.save(depth_path, full_depth)

    if overlay_path is not None:
        overlay = image.copy()
        overlay[full_mask == 0] = (overlay[full_mask == 0] * 0.25).astype(np.uint8)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
    return box, float(crop_mask.mean())


def _select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("mhr", "smplx"), required=True)
    parser.add_argument("--input-root", type=Path, help="MHR camera directory containing images/results/raw")
    parser.add_argument("--dataset-config", type=Path, help="NeuMan/SMPL-X dataset config")
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "predict"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seg-model", default=DEFAULT_SEG_MODEL)
    parser.add_argument("--sapiens2-root", type=Path)
    parser.add_argument("--seg-checkpoint", type=Path)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--no-download-depth", action="store_true")
    parser.add_argument("--mhr-model", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def _build_source(args: argparse.Namespace) -> FrameSource:
    if args.backend == "mhr":
        if args.input_root is None:
            raise ValueError("--input-root is required for --backend mhr")
        return MHRFrameSource(args.input_root, args.mhr_model)
    if args.dataset_config is None:
        raise ValueError("--dataset-config is required for --backend smplx")
    return SMPLXFrameSource(args.dataset_config, args.split)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    if args.margin < 0 or args.stride < 1 or args.start < 0:
        raise ValueError("margin must be non-negative, start non-negative, and stride positive")
    device = _select_device(args.device)
    source = _build_source(args)
    segmenter = Sapiens2Segmenter(
        args.seg_model, device, args.sapiens2_root, args.seg_checkpoint
    )
    depth_checkpoint = _resolve_depth_checkpoint(args.depth_checkpoint, not args.no_download_depth)
    depth_model = SapiensDepth(depth_checkpoint, device)

    mask_dir = args.output_root.expanduser().resolve() / "masks"
    depth_dir = args.output_root.expanduser().resolve() / "sapiens_depth"
    frames = list(source.frames())
    selected = frames[args.start:args.end:args.stride]
    if not selected:
        raise RuntimeError("no frames selected")
    LOGGER.info("Processing %d frame(s)", len(selected))
    for index, frame in enumerate(selected, start=1):
        mask_path = mask_dir / f"{frame.frame_id}.png"
        depth_path = depth_dir / f"{frame.frame_id}.npy"
        if mask_path.exists() and depth_path.exists() and not args.overwrite:
            LOGGER.info("[%d/%d] skip %s", index, len(selected), frame.frame_id)
            continue
        overlay_path = None if args.overlay_dir is None else args.overlay_dir / f"{frame.frame_id}.jpg"
        box, area = process_frame(
            frame, segmenter, depth_model, mask_path, depth_path, args.margin, overlay_path
        )
        LOGGER.info(
            "[%d/%d] %s crop=(%d,%d)-(%d,%d) mask_area=%.4f",
            index, len(selected), frame.frame_id, *box, area,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

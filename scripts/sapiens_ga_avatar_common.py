"""Shared NeuMan/MHR frame and crop handling for the two Sapiens passes.

This module intentionally contains no Sapiens model imports.  It is safe to
use from either the Sapiens2 segmentation environment or the PyTorch Export
depth environment.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch


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
            raise ValueError(f"{raw_path} has no saved vertices; pass --mhr-model to recompute them")
        if not torch.cuda.is_available():
            raise RuntimeError("MHR vertex recomputation requires CUDA")
        from src.body_models import mhr_lbs

        device = "cuda"
        model = mhr_lbs.load_mhr(str(self.mhr_model_path), device=device)
        shape = torch.from_numpy(np.asarray(raw["shape_params"], dtype=np.float32)).unsqueeze(0)
        pose = torch.from_numpy(np.asarray(raw["mhr_model_params"], dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            output = mhr_lbs.mhr_query(model, shape.to(device), pose.to(device), device=device)
        return output["verts"][0].detach().cpu().numpy().astype(np.float32)


class SMPLXFrameSource:
    def __init__(self, dataset_config: Path, split: str, dataset_root: Path | None = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NeuMan SMPL-X frame preparation requires CUDA")
        from omegaconf import OmegaConf
        from src.dataset.neuman_smplx import NeumanSMPLXDataset

        config = OmegaConf.load(str(dataset_config))
        cfg = config.dataset
        if dataset_root is not None:
            cfg.root_dir = str(dataset_root.expanduser().resolve())
        prep_subdivisions = os.environ.get("SAPIENS_PREP_SUBDIVISIONS")
        if prep_subdivisions is not None:
            cfg.body_model.n_subdivisions = int(prep_subdivisions)
        cfg.preload = False
        cfg.data_device = "cuda"
        self.dataset = NeumanSMPLXDataset(cfg, split=split)
        self.body_model = self.dataset.body_model
        self.body_model.cuda()

    def frames(self) -> Iterator[FrameGeometry]:
        from src.body_models.base import skin_vertices

        with torch.inference_mode():
            rest_device = next(self.body_model.parameters()).device
            for frame_index, frame_id in enumerate(self.dataset.frame_ids):
                raw = self.dataset.load_raw(frame_index)
                _, bone_world = self.dataset._world_bone_transforms(frame_id)
                vertices_world = skin_vertices(self.body_model, bone_world.to(rest_device))
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


def build_source(
    backend: str,
    input_root: Path | None,
    dataset_config: Path | None,
    dataset_root: Path | None,
    split: str,
    mhr_model: Path | None,
) -> FrameSource:
    if backend == "mhr":
        if input_root is None:
            raise ValueError("--input-root is required for --backend mhr")
        return MHRFrameSource(input_root, mhr_model)
    if dataset_config is None:
        raise ValueError("--dataset-config is required for --backend smplx")
    return SMPLXFrameSource(dataset_config, split, dataset_root)


def selected_frames(source: FrameSource, start: int, end: int | None, stride: int) -> list[FrameGeometry]:
    if start < 0 or stride < 1:
        raise ValueError("start must be non-negative and stride must be positive")
    frames = list(source.frames())
    selected = frames[start:end:stride]
    if not selected:
        raise RuntimeError("no frames selected")
    return selected


def frame_crop(frame: FrameGeometry, margin: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {frame.image_path}")
    projected = project_vertices(frame.vertices_camera, frame.fx, frame.fy, frame.cx, frame.cy)
    return image, expanded_crop_box(projected, image.shape[1], image.shape[0], margin)


def write_crop_metadata(path: Path, frame: FrameGeometry, image_shape: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "frame_id": frame.frame_id,
                "image_height": image_shape[0],
                "image_width": image_shape[1],
                "crop_box_xyxy": list(box),
            },
            indent=2,
        )
        + "\n"
    )


def read_crop_metadata(path: Path, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"crop metadata not found: {path}; run prepare_sapiens_masks.py first"
        )
    metadata = json.loads(path.read_text())
    expected_shape = (int(metadata["image_height"]), int(metadata["image_width"]))
    if expected_shape != image_shape:
        raise ValueError(f"image shape changed for {path.stem}: metadata={expected_shape}, image={image_shape}")
    box = tuple(int(value) for value in metadata["crop_box_xyxy"])
    if len(box) != 4:
        raise ValueError(f"invalid crop box in {path}: {box}")
    x0, y0, x1, y1 = box
    height, width = image_shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop box outside image in {path}: {box}, image={image_shape}")
    return box


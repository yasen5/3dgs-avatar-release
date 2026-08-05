#!/usr/bin/env python3
"""Dry-run selection of MHR frames whose hands are not occluded by the body.

For every hand vertex, the script tests its camera ray against every non-hand
mesh face.  A frame is valid only when every hand vertex is in the image and
no non-hand face is closer along that camera ray.  Hand vertices are derived
from the MHR skinning weights, using the two palm/finger subtrees rooted at
joints 42 and 78.

The dataset is never modified.  The output directory receives ``split.json``,
``valid.txt``, ``invalid.txt``, and a diagnostic ``panel.png``.

Example::

    python scripts/select_unoccluded_hands.py \
        --input-root /path/to/CoreView_377/Camera_B1 \
        --mhr-model /path/to/mhr_model.pt \
        --output-dir output/hand_visibility
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import numpy.typing as npt
import torch
from src.body_models import mhr_lbs


# These are the complete palm/finger subtrees in MHR's 127-joint hierarchy.
# The wrist/forearm joints immediately preceding them are 41 and 77.
LEFT_HAND_JOINTS = tuple(range(42, 65))
RIGHT_HAND_JOINTS = tuple(range(78, 101))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class FrameResult:
    frame_id: str
    valid: bool
    hand_vertices: int
    occluded_vertices: int
    outside_vertices: int
    left_occluded_vertices: int
    right_occluded_vertices: int


@dataclass(frozen=True)
class FrameDebug:
    image_path: Path
    projected_hand: npt.NDArray[np.float32]
    occluded: npt.NDArray[np.bool_]
    outside: npt.NDArray[np.bool_]
    left_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "MHR camera directory containing images/ and results/raw/. A ZJU "
            "subject directory is also accepted together with --camera-name."
        ),
    )
    parser.add_argument(
        "--camera-name",
        default="Camera_B1",
        help="Camera subdirectory when --input-root is a subject (default: Camera_B1).",
    )
    parser.add_argument(
        "--mhr-model",
        type=Path,
        required=True,
        help="MHR TorchScript checkpoint used only to read rig skinning weights.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hand-weight-threshold",
        type=float,
        default=0.5,
        help="Minimum summed hand-joint skinning weight for a hand vertex (default: 0.5).",
    )
    parser.add_argument(
        "--depth-tolerance-mm",
        type=float,
        default=2.0,
        help="Ignore non-hand depth differences no larger than this (default: 2 mm).",
    )
    parser.add_argument(
        "--allow-outside",
        action="store_true",
        help="Do not invalidate a frame when hand vertices project outside the image.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Rasterization device (default: cuda when available).",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Inclusive numeric frame ID.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive numeric frame ID.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap after frame filtering, useful for a quick smoke test.",
    )
    parser.add_argument("--panel-per-class", type=int, default=8)
    parser.add_argument("--panel-tile-width", type=int, default=320)
    args = parser.parse_args()
    if not 0.0 < args.hand_weight_threshold <= 1.0:
        parser.error("--hand-weight-threshold must be in (0, 1]")
    if args.depth_tolerance_mm < 0:
        parser.error("--depth-tolerance-mm must be non-negative")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.panel_per_class < 1 or args.panel_tile_width < 64:
        parser.error("panel dimensions are too small")
    return args


def numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = stem[5:] if stem.startswith("frame") else stem
    try:
        return int(digits), stem
    except ValueError:
        return sys.maxsize, stem


def camera_dir(input_root: Path, camera_name: str) -> Path:
    root = input_root.expanduser().resolve()
    if (root / "results" / "raw").is_dir():
        return root
    candidate = root / camera_name
    if (candidate / "results" / "raw").is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find results/raw under {root} or {candidate}"
    )


def image_for(image_dir: Path, frame_id: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{frame_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No source image for {frame_id} in {image_dir}")


def descendants(parents: npt.NDArray[np.integer], root: int) -> tuple[int, ...]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for joint, parent in enumerate(parents):
            if joint not in result and int(parent) in result:
                result.add(joint)
                changed = True
    return tuple(sorted(result))


def hand_vertex_masks(
    model_path: Path, threshold: float
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    state = mhr_lbs.mhr_state_dict(str(model_path.expanduser().resolve()))
    parents = mhr_lbs.joint_parents(state).cpu().numpy()
    left_joints = descendants(parents, LEFT_HAND_JOINTS[0])
    right_joints = descendants(parents, RIGHT_HAND_JOINTS[0])
    if left_joints != LEFT_HAND_JOINTS or right_joints != RIGHT_HAND_JOINTS:
        raise ValueError(
            "The checkpoint's joint hierarchy does not match the supported MHR rig: "
            f"left={left_joints}, right={right_joints}"
        )
    weights = mhr_lbs.dense_skinning_weights(state).cpu().numpy()
    left = weights[:, left_joints].sum(axis=1) >= threshold
    right = weights[:, right_joints].sum(axis=1) >= threshold
    if not left.any() or not right.any() or np.any(left & right):
        raise ValueError(
            f"Invalid hand masks from {model_path}: left={left.sum()}, "
            f"right={right.sum()}, overlap={(left & right).sum()}"
        )
    return left, right


def project(
    vertices: npt.NDArray[np.float32], focal: float, width: int, height: int
) -> npt.NDArray[np.float32]:
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float32)
    in_front = vertices[:, 2] > 1e-6
    uv[in_front, 0] = focal * vertices[in_front, 0] / vertices[in_front, 2] + width / 2.0
    uv[in_front, 1] = focal * vertices[in_front, 1] / vertices[in_front, 2] + height / 2.0
    return uv


def ray_hits_before_vertex(
    vertex: npt.NDArray[np.float32],
    vertices: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int64],
    tolerance: float,
) -> bool:
    """Moller-Trumbore test for a camera-origin ray ending at ``vertex``.

    The ray direction is deliberately the vertex itself, so its endpoint has
    parameter \(t=1\).  Intersections within ``tolerance`` in camera-space
    depth of that endpoint are treated as the same surface, not an occluder.
    """
    if len(faces) == 0:
        return False
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    direction = np.asarray(vertex, dtype=np.float64)
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = np.abs(determinant) > 1e-10
    if not usable.any():
        return False
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    s = -triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("j,ij->i", direction, q)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    endpoint_limit = 1.0 - tolerance / max(float(vertex[2]), 1e-8)
    hit = usable & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1.0 + 1e-7)
    hit &= (t > 1e-7) & (t < endpoint_limit)
    return bool(hit.any())


def rays_hit_before_vertices(
    query_vertices: npt.NDArray[np.float32],
    vertices: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int64],
    tolerance: float,
    device: torch.device,
    chunk_size: int = 32,
) -> npt.NDArray[np.bool_]:
    """Test camera rays against every face in bounded-memory batches.

    This deliberately has no rasterized broad phase. Projected triangles can
    be subpixel-sized, so pixel sampling cannot conservatively enumerate every
    face crossed by an exact vertex ray.
    """
    if len(query_vertices) == 0 or len(faces) == 0:
        return np.zeros(len(query_vertices), dtype=bool)

    triangles = torch.as_tensor(vertices[faces], dtype=torch.float64, device=device)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    offset = -triangles[:, 0]
    q = torch.cross(offset, edge1, dim=1)
    t_numerator = torch.einsum("ij,ij->i", edge2, q)
    hits = torch.zeros(len(query_vertices), dtype=torch.bool, device=device)

    with torch.no_grad():
        for start in range(0, len(query_vertices), chunk_size):
            stop = min(start + chunk_size, len(query_vertices))
            direction = torch.as_tensor(
                query_vertices[start:stop], dtype=torch.float64, device=device
            )
            h = torch.cross(direction[:, None, :], edge2[None, :, :], dim=2)
            determinant = torch.einsum("fj,rfj->rf", edge1, h)
            usable = determinant.abs() > 1e-10
            inverse = torch.where(usable, determinant.reciprocal(), 0.0)
            u = inverse * torch.einsum("fj,rfj->rf", offset, h)
            v = inverse * torch.einsum("rj,fj->rf", direction, q)
            t = inverse * t_numerator[None, :]
            endpoint_limit = 1.0 - tolerance / direction[:, 2].clamp_min(1e-8)
            hit = usable & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1.0 + 1e-7)
            hit &= (t > 1e-7) & (t < endpoint_limit[:, None])
            hits[start:stop] = hit.any(dim=1)

    return hits.cpu().numpy()


def classify_frame(
    raw_path: Path,
    image_dir: Path,
    left_mask: npt.NDArray[np.bool_],
    right_mask: npt.NDArray[np.bool_],
    depth_tolerance: float,
    allow_outside: bool,
    device: torch.device,
) -> tuple[FrameResult, FrameDebug]:
    with np.load(raw_path, allow_pickle=False) as raw:
        required = {"vertices", "faces", "focal_length", "width", "height"}
        missing = required.difference(raw.files)
        if missing:
            raise KeyError(f"{raw_path} is missing {sorted(missing)}")
        vertices = np.asarray(raw["vertices"], dtype=np.float32)
        translation_key = "pred_cam_t" if "pred_cam_t" in raw.files else "cam_t"
        if translation_key not in raw.files:
            raise KeyError(f"{raw_path} has no pred_cam_t/cam_t")
        vertices = vertices + np.asarray(raw[translation_key], dtype=np.float32).reshape(1, 3)
        faces = np.asarray(raw["faces"], dtype=np.int64)
        focal = float(np.asarray(raw["focal_length"]).reshape(()))
        width = int(np.asarray(raw["width"]).reshape(()))
        height = int(np.asarray(raw["height"]).reshape(()))

    if len(vertices) != len(left_mask) or faces.max(initial=-1) >= len(vertices):
        raise ValueError(f"Mesh topology in {raw_path} does not match the MHR checkpoint")
    image_path = image_for(image_dir, raw_path.stem)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError(
            f"Image/raw dimensions disagree for {raw_path.stem}: "
            f"image={None if image is None else image.shape[:2]}, raw={(height, width)}"
        )

    hand_mask = left_mask | right_mask
    hand_indices = np.concatenate((np.flatnonzero(left_mask), np.flatnonzero(right_mask)))
    left_count = int(left_mask.sum())
    uv = project(vertices, focal, width, height)
    hand_uv = uv[hand_indices]
    hand_z = vertices[hand_indices, 2]
    finite = np.isfinite(hand_uv).all(axis=1)
    px = np.rint(np.nan_to_num(hand_uv[:, 0], nan=-1.0)).astype(np.int64)
    py = np.rint(np.nan_to_num(hand_uv[:, 1], nan=-1.0)).astype(np.int64)
    inside = finite & (hand_z > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    outside = ~inside

    non_hand_faces = faces[~hand_mask[faces].all(axis=1)]
    occluded = np.zeros(len(hand_indices), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    occluded[inside_indices] = rays_hit_before_vertices(
        vertices[hand_indices[inside_indices]],
        vertices,
        non_hand_faces,
        depth_tolerance,
        device,
    )

    invalid = bool(occluded.any() or (outside.any() and not allow_outside))
    result = FrameResult(
        frame_id=raw_path.stem,
        valid=not invalid,
        hand_vertices=len(hand_indices),
        occluded_vertices=int(occluded.sum()),
        outside_vertices=int(outside.sum()),
        left_occluded_vertices=int(occluded[:left_count].sum()),
        right_occluded_vertices=int(occluded[left_count:].sum()),
    )
    debug = FrameDebug(image_path, hand_uv, occluded, outside, left_count)
    return result, debug


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in positions]


def draw_tile(
    result: FrameResult, debug: FrameDebug, tile_width: int
) -> npt.NDArray[np.uint8]:
    image = cv2.imread(str(debug.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(debug.image_path)
    scale = tile_width / image.shape[1]
    tile = cv2.resize(
        image,
        (tile_width, max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    points = np.rint(debug.projected_hand * scale).astype(np.int32)
    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < tile.shape[1] and 0 <= y < tile.shape[0]):
            continue
        if debug.occluded[index]:
            color, radius = (0, 0, 255), 2
        elif index < debug.left_count:
            color, radius = (70, 220, 70), 1
        else:
            color, radius = (255, 190, 30), 1
        cv2.circle(tile, (x, y), radius, color, -1, cv2.LINE_AA)
    status = "VALID" if result.valid else "INVALID"
    status_color = (40, 190, 40) if result.valid else (30, 30, 220)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        tile, f"{status}  {result.frame_id}", (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        f"occluded={result.occluded_vertices}  outside={result.outside_vertices}",
        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA,
    )
    return tile


def write_panel(
    path: Path,
    results: list[FrameResult],
    debug: list[FrameDebug],
    per_class: int,
    tile_width: int,
) -> None:
    valid = evenly_spaced([i for i, result in enumerate(results) if result.valid], per_class)
    invalid = evenly_spaced([i for i, result in enumerate(results) if not result.valid], per_class)
    selected = valid + invalid
    if not selected:
        raise ValueError("No frames were classified")
    columns = min(per_class, max(len(valid), len(invalid)))
    tiles = [draw_tile(results[i], debug[i], tile_width) for i in selected]
    tile_height = max(tile.shape[0] for tile in tiles)
    blank = np.full((tile_height, tile_width, 3), 35, dtype=np.uint8)
    rows: list[npt.NDArray[np.uint8]] = []
    offset = 0
    for indices in (valid, invalid):
        if not indices:
            continue
        row_tiles = tiles[offset : offset + len(indices)]
        offset += len(indices)
        padded = [
            cv2.copyMakeBorder(tile, 0, tile_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35))
            for tile in row_tiles
        ]
        padded.extend(blank.copy() for _ in range(columns - len(padded)))
        rows.append(np.concatenate(padded, axis=1))
    panel = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError(f"Failed to write {path}")


def write_ids(path: Path, ids: Iterable[str]) -> None:
    path.write_text("".join(f"{frame_id}\n" for frame_id in ids))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    cam_dir = camera_dir(args.input_root, args.camera_name)
    raw_dir = cam_dir / "results" / "raw"
    image_dir = cam_dir / "images"
    raw_paths = sorted(raw_dir.glob("*.npz"), key=numeric_key)
    raw_paths = [
        path for path in raw_paths
        if (args.start_frame is None or numeric_key(path)[0] >= args.start_frame)
        and (args.end_frame is None or numeric_key(path)[0] < args.end_frame)
    ][:: args.stride]
    if args.max_frames is not None:
        raw_paths = raw_paths[: args.max_frames]
    if not raw_paths:
        raise FileNotFoundError(f"No matching .npz fits in {raw_dir}")

    left_mask, right_mask = hand_vertex_masks(args.mhr_model, args.hand_weight_threshold)
    print(
        f"Hand mask: {left_mask.sum()} left + {right_mask.sum()} right vertices "
        f"(weight threshold {args.hand_weight_threshold:g})"
    )
    results: list[FrameResult] = []
    debug: list[FrameDebug] = []
    tolerance = args.depth_tolerance_mm / 1000.0
    for index, raw_path in enumerate(raw_paths, start=1):
        result, frame_debug = classify_frame(
            raw_path, image_dir, left_mask, right_mask, tolerance,
            args.allow_outside, device,
        )
        results.append(result)
        debug.append(frame_debug)
        print(
            f"[{index:04d}/{len(raw_paths):04d}] {result.frame_id}: "
            f"{'valid' if result.valid else 'invalid'} "
            f"(occluded={result.occluded_vertices}, outside={result.outside_vertices})"
        )

    valid_ids = [result.frame_id for result in results if result.valid]
    invalid_ids = [result.frame_id for result in results if not result.valid]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(args.output_dir / "valid.txt", valid_ids)
    write_ids(args.output_dir / "invalid.txt", invalid_ids)
    payload = {
        "dry_run": True,
        "input_root": str(cam_dir),
        "mhr_model": str(args.mhr_model.expanduser().resolve()),
        "criterion": {
            "hand_weight_threshold": args.hand_weight_threshold,
            "depth_tolerance_mm": args.depth_tolerance_mm,
            "outside_invalid": not args.allow_outside,
        },
        "valid": valid_ids,
        "invalid": invalid_ids,
        "frames": [asdict(result) for result in results],
    }
    (args.output_dir / "split.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_panel(
        args.output_dir / "panel.png", results, debug,
        args.panel_per_class, args.panel_tile_width,
    )
    print(
        f"Dry-run split: {len(valid_ids)} valid / {len(invalid_ids)} invalid "
        f"({len(results)} total)"
    )
    print(f"Wrote split and panel to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

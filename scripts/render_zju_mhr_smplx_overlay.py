"""Render side-by-side MHR and SMPL-X mesh projection overlays for ZJU-MoCap.

The expected subject layout is::

    CoreView_377/
        Camera_B1/
            images/000000.jpg (or .png)
            results/raw/000000.npz
            results/smplx/smplx_params.npz

The MHR raw files are the native SAM-3D-Body outputs.  Their ``vertices`` are
model-space vertices and ``pred_cam_t`` is the camera-space translation, so the
projected mesh is ``vertices + pred_cam_t``.  If a raw file does not contain
``vertices``, the script recomputes them from ``shape_params`` and
``mhr_model_params`` with the repository's MHR adapter.

SMPL-X parameters are expected to contain ``global_orient``, ``body_pose``,
``betas``, and ``transl``.  Optional jaw, eye, hand, and expression parameters
are used when present and otherwise set to zero.

Example:

    python render_zju_mhr_smplx_overlay.py \
        --subject-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/dataset/zju_mocap/CoreView_377 \
        --smplx-model /mnt/ssd2/lhmpp_checkpoints/models/Damo_XR_Lab--LHMPP-Prior/snapshots/master/human_model_files/smplx/SMPLX_NEUTRAL.npz \
        --output-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/overlays/mhr_vs_smplx \
        --start-frame 0 --end-frame 570 --stride 10

Each output is a horizontal concatenation of the same source image with the
MHR overlay on the left and the SMPL-X overlay on the right.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import numpy.typing as npt
import smplx
import torch

from src.body_models import mhr_lbs


OPTIONAL_SMPLX_FIELDS = {
    "jaw_pose": 3,
    "leye_pose": 3,
    "reye_pose": 3,
    "left_hand_pose": 45,
    "right_hand_pose": 45,
}
OPTIONAL_SMPLX_ALIASES = {
    "jaw_pose": ("jaw_pose",),
    "leye_pose": ("leye_pose",),
    "reye_pose": ("reye_pose",),
    "left_hand_pose": ("left_hand_pose", "lhand_pose"),
    "right_hand_pose": ("right_hand_pose", "rhand_pose"),
    "expression": ("expression", "expr"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render horizontal MHR|SMPL-X projection-fit overlays for ZJU-MoCap."
    )
    parser.add_argument(
        "--subject-dir",
        type=Path,
        required=True,
        help="Subject directory, e.g. .../zju_mocap/CoreView_377.",
    )
    parser.add_argument(
        "--camera-name",
        default="Camera_B1",
        help="ZJU camera directory under the subject (default: Camera_B1).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="MHR raw-fit directory (default: <camera>/results/raw).",
    )
    parser.add_argument(
        "--smplx-params",
        type=Path,
        default=None,
        help="SMPL-X parameter NPZ (default: <camera>/results/smplx/smplx_params.npz).",
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        required=True,
        help="SMPL-X model directory or direct SMPLX_NEUTRAL.npz file.",
    )
    parser.add_argument(
        "--mhr-model",
        type=Path,
        default=None,
        help="MHR TorchScript checkpoint, only needed if raw files lack vertices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the concatenated overlay PNGs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for body-model evaluation (default: cuda when available).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive frame index bound (default: all available frames).",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--vertex-stride",
        type=int,
        default=1,
        help="Draw every Nth mesh vertex (default: 1).",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Projected vertex radius in pixels (default: 1).",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.9,
        help="Mesh-point opacity over the source image (default: 0.9).",
    )
    parser.add_argument(
        "--gender",
        default="neutral",
        choices=("neutral", "male", "female"),
        help="SMPL-X gender/model filename (default: neutral).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end-frame must be greater than --start-frame")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.vertex_stride < 1:
        parser.error("--vertex-stride must be at least 1")
    if args.point_radius < 1:
        parser.error("--point-radius must be at least 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be in [0, 1]")
    return args


def numeric_frame_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric ZJU frame filename, got {path.name}") from exc


def raw_fit_paths(raw_dir: Path) -> List[Path]:
    paths = sorted(raw_dir.glob("*.npz"), key=numeric_frame_id)
    if not paths:
        raise FileNotFoundError(f"No MHR raw fits found in {raw_dir}")
    return paths


def image_path_for(camera_dir: Path, frame_stem: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = camera_dir / "images" / f"{frame_stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No image found for frame {frame_stem} in {camera_dir / 'images'}"
    )


def get_first_field(data: Mapping[str, npt.NDArray[np.floating[Any]]], names: Sequence[str], label: str) -> npt.NDArray[np.floating[Any]]:
    for name in names:
        if name in data:
            return np.asarray(data[name], dtype=np.float32)
    raise KeyError(f"SMPL-X fit is missing {label}; tried {', '.join(names)}")


def frame_array(
    value: npt.NDArray[np.floating[Any]],
    num_frames: int,
    width: int,
    name: str,
) -> npt.NDArray[np.floating[Any]]:
    """Normalize a constant or per-frame vector to ``(num_frames, width)``."""
    value = np.asarray(value, dtype=np.float32)
    if value.ndim >= 2 and value.shape[0] in (1, num_frames):
        value = value.reshape(value.shape[0], -1)
    if value.shape == (width,):
        return np.broadcast_to(value, (num_frames, width)).copy()
    if value.shape == (1, width):
        return np.broadcast_to(value, (num_frames, width)).copy()
    if value.shape == (num_frames, width):
        return value
    raise ValueError(
        f"SMPL-X field {name!r} must have shape ({width},), (1, {width}), "
        f"or ({num_frames}, {width}); got {value.shape}"
    )


def load_smplx_fit(path: Path) -> Tuple[Dict[str, npt.NDArray[np.floating[Any]]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"SMPL-X parameter file not found: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: np.asarray(loaded[name]) for name in loaded.files}

    global_orient = get_first_field(
        data, ("global_orient", "global_rot", "root_pose"), "global orientation"
    )
    if global_orient.ndim == 1:
        num_frames = 1
    elif global_orient.ndim >= 2:
        num_frames = global_orient.shape[0]
    else:
        raise ValueError(f"Unexpected global orientation shape: {global_orient.shape}")

    betas = get_first_field(data, ("betas", "shape"), "shape coefficients")
    betas_flat = betas.reshape(-1) if betas.ndim == 1 else betas.reshape(betas.shape[0], -1)
    num_betas = betas_flat.shape[-1]

    fields: Dict[str, npt.NDArray[np.floating[Any]]] = {
        "global_orient": frame_array(global_orient, num_frames, 3, "global_orient"),
        "body_pose": frame_array(
            get_first_field(data, ("body_pose",), "body pose"),
            num_frames,
            63,
            "body_pose",
        ),
        "betas": frame_array(
            betas,
            num_frames,
            num_betas,
            "betas",
        ),
        "transl": frame_array(
            get_first_field(data, ("transl", "translation", "trans"), "translation"),
            num_frames,
            3,
            "transl",
        ),
    }

    for name, width in OPTIONAL_SMPLX_FIELDS.items():
        value = next((data[alias] for alias in OPTIONAL_SMPLX_ALIASES[name] if alias in data), None)
        if value is not None:
            fields[name] = frame_array(value, num_frames, width, name)
        else:
            fields[name] = np.zeros((num_frames, width), dtype=np.float32)

    expression = next((data[alias] for alias in OPTIONAL_SMPLX_ALIASES["expression"] if alias in data), None)
    if expression is None:
        fields["expression"] = np.zeros((num_frames, 10), dtype=np.float32)
    else:
        expression = np.asarray(expression, dtype=np.float32)
        if expression.ndim == 1:
            expression_width = expression.shape[0]
        else:
            expression_width = expression.reshape(expression.shape[0], -1).shape[1]
        fields["expression"] = frame_array(expression, num_frames, expression_width, "expression")

    if not np.isfinite(np.concatenate(list(fields.values()), axis=1)).all():
        raise ValueError(f"SMPL-X parameter file contains non-finite values: {path}")
    return fields, num_frames


def resolve_smplx_model(path: Path, gender: str) -> Path:
    """Resolve a model directory to the concrete file accepted by smplx.create."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"SMPL-X model path not found: {path}")

    candidates = (
        path / f"SMPLX_{gender.upper()}.npz",
        path / "smplx" / f"SMPLX_{gender.upper()}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find an SMPL-X model; tried {tried}")


def make_smplx_model(
    model_path: Path,
    gender: str,
    num_betas: int,
    num_expression_coeffs: int,
    device: torch.device,
) -> torch.nn.Module:
    model_file = resolve_smplx_model(model_path, gender)
    model: torch.nn.Module = smplx.create(
        model_path=str(model_file),
        model_type="smplx",
        gender=gender,
        ext=model_file.suffix.lstrip("."),
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        use_pca=False,
        batch_size=1,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def project_points(vertices: npt.NDArray[np.floating[Any]], focal: float, width: int, height: int) -> npt.NDArray[np.floating[Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3), got {vertices.shape}")
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-6)
    uv = np.empty((vertices.shape[0], 2), dtype=np.float32)
    uv[:] = np.nan
    uv[valid, 0] = focal * vertices[valid, 0] / z[valid] + width / 2.0
    uv[valid, 1] = focal * vertices[valid, 1] / z[valid] + height / 2.0
    return uv


def draw_projected_vertices(
    image: npt.NDArray[np.uint8],
    vertices: npt.NDArray[np.floating[Any]],
    focal: float,
    color: Tuple[int, int, int],
    vertex_stride: int,
    point_radius: int,
    alpha: float,
) -> npt.NDArray[np.uint8]:
    height, width = image.shape[:2]
    uv = project_points(vertices[::vertex_stride], focal, width, height)
    valid = np.isfinite(uv).all(axis=1)
    uv = np.rint(uv[valid]).astype(np.int32)
    if uv.size:
        in_image = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        uv = uv[in_image]

    overlay = image.copy()
    if uv.size:
        if point_radius == 1:
            overlay[uv[:, 1], uv[:, 0]] = color
        else:
            for x, y in uv:
                cv2.circle(overlay, (int(x), int(y)), point_radius, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def add_label(image: npt.NDArray[np.uint8], text: str) -> npt.NDArray[np.uint8]:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def mhr_vertices_from_raw(
    raw: Mapping[str, npt.NDArray[np.floating[Any]]],
    mhr_model: torch.jit.ScriptModule | None,
    device: torch.device,
) -> npt.NDArray[np.floating[Any]]:
    if "vertices" in raw:
        vertices = np.asarray(raw["vertices"], dtype=np.float32)
    else:
        if mhr_model is None:
            raise ValueError(
                "MHR raw fit has no 'vertices'. Supply --mhr-model so the mesh "
                "can be recomputed from shape_params and mhr_model_params."
            )
        for field in ("shape_params", "mhr_model_params"):
            if field not in raw:
                raise KeyError(f"MHR raw fit is missing {field!r}")
        shape = torch.from_numpy(np.asarray(raw["shape_params"], dtype=np.float32)).to(device)
        pose = torch.from_numpy(np.asarray(raw["mhr_model_params"], dtype=np.float32)).to(device)
        with torch.inference_mode():
            output = mhr_lbs.mhr_query(
                mhr_model,
                shape.unsqueeze(0),
                pose.unsqueeze(0),
                device=str(device),
            )
        vertices = output["verts"][0].detach().cpu().numpy().astype(np.float32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected MHR vertex shape: {vertices.shape}")
    translation_name = "pred_cam_t" if "pred_cam_t" in raw else "cam_t"
    if translation_name not in raw:
        raise KeyError("MHR raw fit is missing 'pred_cam_t' (or legacy 'cam_t')")
    translation = np.asarray(raw[translation_name], dtype=np.float32).reshape(3)
    return vertices + translation[None, :]


def load_raw(path: Path) -> Dict[str, npt.NDArray[np.floating[Any]]]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def selected_paths(paths: Iterable[Path], start: int, end: int | None, stride: int) -> List[Path]:
    selected = []
    for path in paths:
        frame_id = numeric_frame_id(path)
        if frame_id < start or (end is not None and frame_id >= end):
            continue
        if (frame_id - start) % stride == 0:
            selected.append(path)
    if not selected:
        bound = f"[{start}, {end})" if end is not None else f">={start}"
        raise ValueError(f"No MHR frames selected in frame range {bound}")
    return selected


def render(args: argparse.Namespace) -> int:
    subject_dir = args.subject_dir
    camera_dir = subject_dir / args.camera_name
    raw_dir = args.raw_dir or camera_dir / "results" / "raw"
    smplx_params_path = args.smplx_params or camera_dir / "results" / "smplx" / "smplx_params.npz"

    all_raw_paths = raw_fit_paths(raw_dir)
    paths = selected_paths(all_raw_paths, args.start_frame, args.end_frame, args.stride)
    smplx_fields, num_smplx_frames = load_smplx_fit(smplx_params_path)
    max_frame = max(numeric_frame_id(path) for path in paths)
    if max_frame >= num_smplx_frames:
        raise ValueError(
            f"SMPL-X fit has {num_smplx_frames} frames but frame {max_frame} was requested"
        )

    device = torch.device(args.device)
    model = make_smplx_model(
        args.smplx_model,
        args.gender,
        smplx_fields["betas"].shape[1],
        smplx_fields["expression"].shape[1],
        device,
    )
    mhr_model = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for batch_start in range(0, len(paths), 16):
        batch_paths = paths[batch_start : batch_start + 16]
        frame_indices = [numeric_frame_id(path) for path in batch_paths]
        model_inputs = {
            name: torch.from_numpy(values[frame_indices]).to(device)
            for name, values in smplx_fields.items()
        }
        with torch.inference_mode():
            smplx_output = model(return_verts=True, **model_inputs)
        smplx_vertices = smplx_output.vertices.detach().cpu().numpy()

        for local_idx, (raw_path, frame_idx) in enumerate(zip(batch_paths, frame_indices)):
            frame_stem = raw_path.stem
            output_path = args.output_dir / f"{frame_stem}_mhr_vs_smplx.png"
            if output_path.exists() and not args.overwrite:
                continue

            raw = load_raw(raw_path)
            image_path = image_path_for(camera_dir, frame_stem)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            if "focal_length" not in raw:
                raise KeyError(f"MHR raw fit is missing 'focal_length': {raw_path}")
            focal = float(np.asarray(raw["focal_length"]).reshape(()))

            if "vertices" not in raw:
                if mhr_model is None:
                    if args.mhr_model is None:
                        raise ValueError(
                            f"{raw_path} has no saved vertices; supply --mhr-model"
                        )
                    mhr_model = mhr_lbs.load_mhr(str(args.mhr_model), device=str(device))
            mhr_vertices = mhr_vertices_from_raw(raw, mhr_model, device)

            mhr_panel = draw_projected_vertices(
                image,
                mhr_vertices,
                focal,
                (0, 255, 0),
                args.vertex_stride,
                args.point_radius,
                args.overlay_alpha,
            )
            smplx_panel = draw_projected_vertices(
                image,
                smplx_vertices[local_idx],
                focal,
                (0, 255, 0),
                args.vertex_stride,
                args.point_radius,
                args.overlay_alpha,
            )
            mhr_panel = add_label(mhr_panel, f"MHR projection fit  frame {frame_stem}")
            smplx_panel = add_label(smplx_panel, f"SMPL-X projection fit  frame {frame_stem}")
            concatenated = np.concatenate((mhr_panel, smplx_panel), axis=1)
            if not cv2.imwrite(str(output_path), concatenated):
                raise RuntimeError(f"Failed to write overlay: {output_path}")
            written += 1

    print(f"Wrote {written} MHR|SMPL-X overlay images to {args.output_dir}")
    return 0


def main() -> int:
    return render(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

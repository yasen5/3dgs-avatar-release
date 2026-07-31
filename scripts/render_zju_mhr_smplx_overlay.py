"""Render side-by-side MHR and SMPL mesh projection overlays for ZJU-MoCap.

The expected subject layout is::

    CoreView_377/
        Camera_B1/
            images/000000.jpg (or .png)
            results/raw/000000.npz
        annots.npy
        new_params/0.npy
        new_vertices/0.npy

The MHR raw files are the native SAM-3D-Body outputs.  Their ``vertices`` are
model-space vertices and ``pred_cam_t`` is the camera-space translation, so the
projected mesh is ``vertices + pred_cam_t``.  If a raw file does not contain
``vertices``, the script recomputes them from ``shape_params`` and
``mhr_model_params`` with the repository's MHR adapter.

The native ZJU files in ``new_params`` are dictionaries containing ``Rh``,
``Th``, ``poses``, and ``shapes``.  They are the standard 72-parameter SMPL
fits: root orientation, 69 body-pose values, translation, and 10 shape
coefficients.  ``Th`` is in the ZJU world coordinate system and is transformed
with the selected camera from ``annots.npy`` before projection.

Example:

    python render_zju_mhr_smplx_overlay.py \
        --subject-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/dataset/zju_mocap/CoreView_377 \
        --smpl-vertices-dir /mnt/ssd2/data/other/3d-human-datasets/zju_mocap_raw/CoreView_377/new_vertices \
        --output-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/overlays/mhr_vs_smpl \
        --start-frame 0 --end-frame 570 --stride 10

Each output is a horizontal concatenation of the same source image with the
MHR overlay on the left and the SMPL overlay on the right.
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
import torch

from src.body_models import mhr_lbs


Projection = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render horizontal MHR|SMPL projection-fit overlays for ZJU-MoCap."
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
        "--smpl-params-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing native ZJU SMPL files such as new_params/0.npy. "
            "If omitted, searches the subject and resolved image-source directories."
        ),
    )
    parser.add_argument(
        "--smpl-vertices-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing precomputed posed SMPL vertices such as "
            "new_vertices/0.npy. If omitted, searches the subject and resolved "
            "image-source directories."
        ),
    )
    parser.add_argument(
        "--smpl-model",
        type=Path,
        default=None,
        help="Optional SMPL model for fallback evaluation when vertices are unavailable.",
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
        help="SMPL gender/model filename (default: neutral).",
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


def resolve_smpl_params_dir(
    subject_dir: Path,
    camera_dir: Path,
    requested: Path | None,
) -> Path:
    """Find native ZJU ``new_params``/``params`` files for this subject."""
    if requested is not None:
        if not requested.is_dir():
            raise FileNotFoundError(f"SMPL parameter directory not found: {requested}")
        return requested

    candidates = [subject_dir / "new_params", subject_dir / "params"]

    # The prepared ZJU images are symlinks to the original release.  Use that
    # source subject as a fallback so the prepared MHR layout can be rendered
    # without copying the original per-frame SMPL files.
    image_dir = camera_dir / "images"
    for image_path in sorted(image_dir.glob("*")):
        try:
            source_image = image_path.resolve()
        except OSError:
            continue
        source_camera = source_image.parent
        if source_camera.name.startswith("Camera_B"):
            source_subject = source_camera.parent
            candidates.extend((source_subject / "new_params", source_subject / "params"))
        break

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find native ZJU SMPL parameters; tried {tried}")


def resolve_smpl_vertices_dir(
    subject_dir: Path,
    camera_dir: Path,
    requested: Path | None,
) -> Path | None:
    """Find precomputed posed SMPL vertices, if available."""
    if requested is not None:
        if not requested.is_dir():
            raise FileNotFoundError(f"SMPL vertex directory not found: {requested}")
        return requested

    candidates = [subject_dir / "new_vertices", subject_dir / "vertices"]
    image_dir = camera_dir / "images"
    for image_path in sorted(image_dir.glob("*")):
        try:
            source_image = image_path.resolve()
        except OSError:
            continue
        source_camera = source_image.parent
        if source_camera.name.startswith("Camera_B"):
            source_subject = source_camera.parent
            candidates.extend((source_subject / "new_vertices", source_subject / "vertices"))
        break

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def smpl_param_paths(params_dir: Path) -> Dict[int, Path]:
    paths = sorted(params_dir.glob("*.npy"), key=numeric_frame_id)
    if not paths:
        raise FileNotFoundError(f"No ZJU SMPL parameter files found in {params_dir}")
    result = {numeric_frame_id(path): path for path in paths}
    if len(result) != len(paths):
        raise ValueError(f"Duplicate numeric frame names in {params_dir}")
    return result


def load_smpl_vertices(
    vertices_dir: Path,
    frame_indices: Sequence[int],
) -> npt.NDArray[np.floating[Any]]:
    paths = smpl_param_paths(vertices_dir)
    vertices = []
    for frame_index in frame_indices:
        path = paths.get(frame_index)
        if path is None:
            raise FileNotFoundError(
                f"No precomputed SMPL vertices for frame {frame_index} in {vertices_dir}"
            )
        value = np.asarray(np.load(path), dtype=np.float32)
        if value.shape != (6890, 3):
            raise ValueError(f"Expected SMPL vertices with shape (6890, 3) in {path}, got {value.shape}")
        vertices.append(value)
    result = np.stack(vertices, axis=0)
    if not np.isfinite(result).all():
        raise ValueError(f"Precomputed SMPL vertices contain non-finite values: {vertices_dir}")
    return result


def load_smpl_fit(
    params_dir: Path,
    frame_indices: Sequence[int],
) -> Dict[str, npt.NDArray[np.floating[Any]]]:
    """Load native ZJU SMPL dictionaries in the order requested by the MHR fits."""
    paths = smpl_param_paths(params_dir)
    records: List[Dict[str, npt.NDArray[np.floating[Any]]]] = []
    for frame_index in frame_indices:
        path = paths.get(frame_index)
        if path is None:
            raise FileNotFoundError(
                f"No ZJU SMPL parameter for frame {frame_index} in {params_dir}"
            )
        loaded = np.load(path, allow_pickle=True)
        if not isinstance(loaded, np.ndarray) or loaded.shape != () or loaded.dtype != object:
            raise ValueError(f"Expected a pickled parameter dictionary in {path}")
        data = loaded.item()
        if not isinstance(data, dict):
            raise ValueError(f"Expected a parameter dictionary in {path}")

        missing = [name for name in ("Rh", "Th", "poses", "shapes") if name not in data]
        if missing:
            raise KeyError(f"ZJU SMPL file {path} is missing {missing}")
        root_orient = np.asarray(data["Rh"], dtype=np.float32).reshape(-1)
        transl = np.asarray(data["Th"], dtype=np.float32).reshape(-1)
        poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1)
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)
        if root_orient.shape != (3,) or transl.shape != (3,) or poses.shape != (72,):
            raise ValueError(
                f"Unexpected SMPL shapes in {path}: Rh={root_orient.shape}, "
                f"Th={transl.shape}, poses={poses.shape}"
            )
        records.append(
            {
                "global_orient": root_orient,
                # smplx.SMPL expects all 23 non-root joints (69 values).
                "body_pose": poses[3:],
                "betas": betas,
                "transl": transl,
            }
        )

    fields = {
        name: np.stack([record[name] for record in records], axis=0)
        for name in records[0]
    }
    if not np.isfinite(np.concatenate(list(fields.values()), axis=1)).all():
        raise ValueError(f"ZJU SMPL parameters contain non-finite values: {params_dir}")
    return fields


def resolve_smpl_model(path: Path, gender: str) -> Path:
    """Resolve a model directory to the concrete file accepted by smplx.create."""
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"SMPL model path not found: {path}")

    candidates = (
        path / f"SMPL_{gender.upper()}.pkl",
        path / "smpl" / f"SMPL_{gender.upper()}.pkl",
        path / f"SMPL_{gender.upper()}.npz",
        path / "smpl" / f"SMPL_{gender.upper()}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find an SMPL model; tried {tried}")


def make_smpl_model(
    model_path: Path,
    gender: str,
    num_betas: int,
    device: torch.device,
) -> torch.nn.Module:
    import smplx

    model_file = resolve_smpl_model(model_path, gender)
    model: torch.nn.Module = smplx.create(
        model_path=str(model_file),
        model_type="smpl",
        gender=gender,
        ext=model_file.suffix.lstrip("."),
        num_betas=num_betas,
        batch_size=1,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_zju_camera(
    subject_dir: Path,
    camera_name: str,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Load ``R``, ``T``, ``K``, and ``D`` for a ZJU camera."""
    annots_path = subject_dir / "annots.npy"
    if not annots_path.is_file():
        raise FileNotFoundError(f"ZJU annotation file not found: {annots_path}")
    annots = np.load(annots_path, allow_pickle=True).item()
    if not camera_name.startswith("Camera_B"):
        raise ValueError(f"Expected a ZJU camera such as Camera_B1, got {camera_name!r}")
    try:
        camera_index = int(camera_name.removeprefix("Camera_B")) - 1
    except ValueError as exc:
        raise ValueError(f"Expected a ZJU camera such as Camera_B1, got {camera_name!r}") from exc

    camera_names = {
        Path(path).parent.name
        for path in annots["ims"][0]["ims"]
    }
    if camera_name not in camera_names:
        raise ValueError(
            f"{camera_name} is not present in {annots_path}; "
            f"available cameras: {sorted(camera_names)}"
        )
    cams = annots["cams"]
    if camera_index < 0 or camera_index >= len(cams["K"]):
        raise ValueError(f"Camera index out of range for {camera_name}: {camera_index}")

    K = np.asarray(cams["K"][camera_index], dtype=np.float32)
    D = np.asarray(cams["D"][camera_index], dtype=np.float32).reshape(-1)
    R = np.asarray(cams["R"][camera_index], dtype=np.float32)
    # ZJU stores T in millimetres; SMPL vertices and MHR translations are in m.
    T = np.asarray(cams["T"][camera_index], dtype=np.float32).reshape(3) / 1000.0
    if K.shape != (3, 3) or R.shape != (3, 3):
        raise ValueError(f"Unexpected camera shapes in {annots_path}: K={K.shape}, R={R.shape}")
    return R, T, K, D


def project_points(
    vertices: npt.NDArray[np.floating[Any]],
    projection: Projection,
    distortion: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.floating[Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (N, 3), got {vertices.shape}")
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-6)
    uv = np.empty((vertices.shape[0], 2), dtype=np.float32)
    uv[:] = np.nan
    fx, fy, cx, cy = projection
    if distortion is None or not np.any(np.abs(distortion) > 1e-12):
        uv[valid, 0] = fx * vertices[valid, 0] / z[valid] + cx
        uv[valid, 1] = fy * vertices[valid, 1] / z[valid] + cy
    else:
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        projected, _ = cv2.projectPoints(
            vertices[valid].reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            camera_matrix,
            np.asarray(distortion, dtype=np.float32),
        )
        uv[valid] = projected.reshape(-1, 2)
    return uv


def draw_projected_vertices(
    image: npt.NDArray[np.uint8],
    vertices: npt.NDArray[np.floating[Any]],
    projection: Projection,
    color: Tuple[int, int, int],
    vertex_stride: int,
    point_radius: int,
    alpha: float,
    distortion: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.uint8]:
    height, width = image.shape[:2]
    uv = project_points(vertices[::vertex_stride], projection, distortion)
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

    all_raw_paths = raw_fit_paths(raw_dir)
    paths = selected_paths(all_raw_paths, args.start_frame, args.end_frame, args.stride)
    frame_indices = [numeric_frame_id(path) for path in paths]
    smpl_vertices_dir = resolve_smpl_vertices_dir(
        subject_dir,
        camera_dir,
        args.smpl_vertices_dir,
    )
    smpl_vertices_world_all = (
        load_smpl_vertices(smpl_vertices_dir, frame_indices)
        if smpl_vertices_dir is not None
        else None
    )
    camera_R, camera_T, camera_K, camera_D = load_zju_camera(subject_dir, args.camera_name)

    device = torch.device(args.device)
    model = None
    smpl_fields = None
    if smpl_vertices_world_all is None:
        if args.smpl_model is None:
            raise FileNotFoundError(
                "No precomputed SMPL vertices found. Supply --smpl-vertices-dir "
                "or --smpl-model together with --smpl-params-dir."
            )
        smpl_params_dir = resolve_smpl_params_dir(
            subject_dir,
            camera_dir,
            args.smpl_params_dir,
        )
        smpl_fields = load_smpl_fit(smpl_params_dir, frame_indices)
        model = make_smpl_model(
            args.smpl_model,
            args.gender,
            smpl_fields["betas"].shape[1],
            device,
        )
    mhr_model = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for batch_start in range(0, len(paths), 16):
        batch_paths = paths[batch_start : batch_start + 16]
        batch_end = batch_start + len(batch_paths)
        if smpl_vertices_world_all is not None:
            smpl_vertices_world = smpl_vertices_world_all[batch_start:batch_end]
        else:
            assert model is not None and smpl_fields is not None
            model_inputs = {
                name: torch.from_numpy(values[batch_start:batch_end]).to(device)
                for name, values in smpl_fields.items()
            }
            with torch.inference_mode():
                smpl_output = model(return_verts=True, **model_inputs)
            smpl_vertices_world = smpl_output.vertices.detach().cpu().numpy()
        smpl_vertices = np.einsum(
            "ij,bvj->bvi",
            camera_R,
            smpl_vertices_world,
        ) + camera_T[None, None, :]

        for local_idx, raw_path in enumerate(batch_paths):
            frame_stem = raw_path.stem
            output_path = args.output_dir / f"{frame_stem}_mhr_vs_smpl.png"
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
            mhr_projection = (focal, focal, width / 2.0, height / 2.0)
            smpl_projection = (
                float(camera_K[0, 0]),
                float(camera_K[1, 1]),
                float(camera_K[0, 2]),
                float(camera_K[1, 2]),
            )

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
                mhr_projection,
                (0, 255, 0),
                args.vertex_stride,
                args.point_radius,
                args.overlay_alpha,
            )
            smpl_panel = draw_projected_vertices(
                image,
                smpl_vertices[local_idx],
                smpl_projection,
                (0, 255, 0),
                args.vertex_stride,
                args.point_radius,
                args.overlay_alpha,
                camera_D,
            )
            mhr_panel = add_label(mhr_panel, f"MHR projection fit  frame {frame_stem}")
            smpl_panel = add_label(smpl_panel, f"SMPL projection fit  frame {frame_stem}")
            concatenated = np.concatenate((mhr_panel, smpl_panel), axis=1)
            if not cv2.imwrite(str(output_path), concatenated):
                raise RuntimeError(f"Failed to write overlay: {output_path}")
            written += 1

    print(f"Wrote {written} MHR|SMPL overlay images to {args.output_dir}")
    return 0


def main() -> int:
    return render(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

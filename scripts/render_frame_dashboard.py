"""Render a per-frame dashboard of the resources used to train on a ZJU-MoCap subject.

For each selected frame this produces a single 2x2 panel image:

    +----------------------+----------------------+
    | source image          | MHR projection fit   |
    +----------------------+----------------------+
    | mask (e.g. Sapiens2)   | masked image          |
    +----------------------+----------------------+

With ``--render-smplx`` the SMPL projection-fit overlay (see
``render_zju_mhr_smplx_overlay.py``) is added as a third panel in the top
row, padding the bottom row to match:

    +--------------+--------------+--------------+
    | source image  | MHR overlay   | SMPL overlay |
    +--------------+--------------+--------------+
    | mask           | masked image  | (blank)      |
    +--------------+--------------+--------------+

This requires the ZJU-MoCap SMPL layout (``annots.npy`` and
``new_params``/``new_vertices``), so it defaults to off and only works for
subjects laid out like ZJU-MoCap.

The expected subject layout (matching ``render_zju_mhr_smplx_overlay.py`` and
``MHRNativeDataset``) is::

    CoreView_377/
        annots.npy
        Camera_B1/
            images/000000.jpg (or .png)
            masks/000000.png
            results/raw/000000.npz

Example:

    python render_frame_dashboard.py \
        --subject-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/dataset/zju_mocap/CoreView_377 \
        --output-dir /mnt/ssd2/better_rigs_runs/zju_377_prep/dashboards/zju_377 \
        --start-frame 0 --end-frame 570 --stride 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import numpy.typing as npt
import torch

from src.body_models import mhr_lbs
from scripts.render_zju_mhr_smplx_overlay import (
    add_label,
    draw_projected_vertices,
    load_raw,
    load_smpl_fit,
    load_smpl_vertices,
    load_zju_camera,
    make_smpl_model,
    mhr_vertices_from_raw,
    numeric_frame_id,
    raw_fit_paths,
    resolve_smpl_params_dir,
    resolve_smpl_vertices_dir,
    selected_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a per-frame dashboard (image / MHR overlay / mask / masked image)."
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
        help="Camera directory under the subject (default: Camera_B1).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="MHR raw-fit directory (default: <camera>/results/raw).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Mask directory (default: <camera>/masks, i.e. the Sapiens2 masks).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Image directory (default: <camera>/images). Set this together with "
            "--raw-dir and --mask-dir to point at a video-derived prepare_mhr_"
            "dense_track.py / prepare_mhr_keyframe_track.py run's pre-split "
            "intermediates (<out-dir>/frames, /raw_sam3d, /sapiens2_masks) "
            "instead of a ZJU-MoCap camera directory."
        ),
    )
    parser.add_argument(
        "--mhr-model",
        type=Path,
        default=None,
        help="MHR TorchScript checkpoint, only needed if raw files lack 'vertices'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the dashboard PNGs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for MHR evaluation (default: cuda when available).",
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
        help="Draw every Nth MHR mesh vertex (default: 1).",
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
        help="MHR mesh-point opacity over the source image (default: 0.9).",
    )
    parser.add_argument(
        "--panel-scale",
        type=float,
        default=1.0,
        help="Resize factor applied to each panel before assembling the grid (default: 1.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    parser.add_argument(
        "--render-smplx",
        action="store_true",
        default=False,
        help=(
            "Add a third top-row panel with the SMPL projection-fit overlay "
            "(default: False). Requires the ZJU-MoCap SMPL layout "
            "(annots.npy and new_params/new_vertices)."
        ),
    )
    parser.add_argument(
        "--smpl-params-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing native ZJU SMPL files such as new_params/0.npy. "
            "Only used with --render-smplx; if omitted, searches the subject and "
            "resolved image-source directories."
        ),
    )
    parser.add_argument(
        "--smpl-vertices-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing precomputed posed SMPL vertices such as "
            "new_vertices/0.npy. Only used with --render-smplx; if omitted, "
            "searches the subject and resolved image-source directories."
        ),
    )
    parser.add_argument(
        "--smpl-model",
        type=Path,
        default=None,
        help="Optional SMPL model for fallback evaluation when precomputed vertices are unavailable.",
    )
    parser.add_argument(
        "--gender",
        default="neutral",
        choices=("neutral", "male", "female"),
        help="SMPL gender/model filename (default: neutral).",
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
    if args.panel_scale <= 0.0:
        parser.error("--panel-scale must be positive")
    return args


def image_path_for(image_dir: Path, frame_stem: str) -> Path:
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = image_dir / f"{frame_stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for frame {frame_stem} in {image_dir}")


def mask_path_for(mask_dir: Path, frame_stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{frame_stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask found for frame {frame_stem} in {mask_dir}")


def load_mask(path: Path, height: int, width: int) -> npt.NDArray[np.bool_]:
    raw_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    mask = np.asarray(raw_mask)
    if mask.shape != (height, width):
        raise ValueError(
            f"Mask/image size mismatch for {path}: mask={mask.shape}, image=({height}, {width})"
        )
    return mask != 0


def render_mask_panel(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[mask] = (255, 255, 255)
    return panel


def render_masked_image_panel(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    panel = np.zeros_like(image)
    panel[mask] = image[mask]
    return panel


def scale_panel(panel: npt.NDArray[np.uint8], scale: float) -> npt.NDArray[np.uint8]:
    if scale == 1.0:
        return panel
    height, width = panel.shape[:2]
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(panel, new_size, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def assemble_grid(panels: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.uint8]:
    if len(panels) == 4:
        top = np.concatenate((panels[0], panels[1]), axis=1)
        bottom = np.concatenate((panels[2], panels[3]), axis=1)
        return np.concatenate((top, bottom), axis=0)
    if len(panels) == 5:
        # image | MHR overlay | SMPL overlay
        # mask  | masked image | (blank, padded to match panel size)
        blank = np.zeros_like(panels[4])
        top = np.concatenate((panels[0], panels[1], panels[2]), axis=1)
        bottom = np.concatenate((panels[3], panels[4], blank), axis=1)
        return np.concatenate((top, bottom), axis=0)
    raise ValueError(f"Expected 4 or 5 panels, got {len(panels)}")


def render(args: argparse.Namespace) -> int:
    subject_dir = args.subject_dir
    camera_dir = subject_dir / args.camera_name
    raw_dir = args.raw_dir or camera_dir / "results" / "raw"
    mask_dir = args.mask_dir or camera_dir / "masks"
    image_dir = args.image_dir or camera_dir / "images"
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    all_raw_paths = raw_fit_paths(raw_dir)
    paths = selected_paths(all_raw_paths, args.start_frame, args.end_frame, args.stride)

    device = torch.device(args.device)
    mhr_model: Any = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smpl_vertices_by_frame: dict[int, npt.NDArray[np.floating[Any]]] | None = None
    smpl_projection: tuple[float, float, float, float] | None = None
    smpl_distortion: npt.NDArray[np.floating[Any]] | None = None
    if args.render_smplx:
        frame_indices = [numeric_frame_id(path) for path in paths]
        smpl_vertices_dir = resolve_smpl_vertices_dir(
            subject_dir, camera_dir, args.smpl_vertices_dir
        )
        if smpl_vertices_dir is not None:
            smpl_vertices_world_all = load_smpl_vertices(smpl_vertices_dir, frame_indices)
        else:
            if args.smpl_model is None:
                raise FileNotFoundError(
                    "No precomputed SMPL vertices found. Supply --smpl-vertices-dir "
                    "or --smpl-model together with --smpl-params-dir."
                )
            smpl_params_dir = resolve_smpl_params_dir(
                subject_dir, camera_dir, args.smpl_params_dir
            )
            smpl_fields = load_smpl_fit(smpl_params_dir, frame_indices)
            smpl_model = make_smpl_model(
                args.smpl_model, args.gender, smpl_fields["betas"].shape[1], device
            )
            model_inputs = {
                name: torch.from_numpy(values).to(device)
                for name, values in smpl_fields.items()
            }
            with torch.inference_mode():
                smpl_output = smpl_model(return_verts=True, **model_inputs)
            smpl_vertices_world_all = smpl_output.vertices.detach().cpu().numpy()

        camera_R, camera_T, camera_K, camera_D = load_zju_camera(subject_dir, args.camera_name)
        smpl_vertices_camera_all = (
            np.einsum("ij,bvj->bvi", camera_R, smpl_vertices_world_all) + camera_T[None, None, :]
        )
        smpl_vertices_by_frame = {
            frame_index: smpl_vertices_camera_all[i]
            for i, frame_index in enumerate(frame_indices)
        }
        smpl_projection = (
            float(camera_K[0, 0]),
            float(camera_K[1, 1]),
            float(camera_K[0, 2]),
            float(camera_K[1, 2]),
        )
        smpl_distortion = camera_D

    written = 0
    for raw_path in paths:
        frame_stem = raw_path.stem
        output_path = args.output_dir / f"{frame_stem}_dashboard.png"
        if output_path.exists() and not args.overwrite:
            continue

        raw = load_raw(raw_path)
        image_path = image_path_for(image_dir, frame_stem)
        loaded_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded_image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image: npt.NDArray[np.uint8] = np.asarray(loaded_image, dtype=np.uint8)
        height, width = image.shape[:2]

        if "focal_length" not in raw:
            raise KeyError(f"MHR raw fit is missing 'focal_length': {raw_path}")
        focal = float(np.asarray(raw["focal_length"]).reshape(()))
        mhr_projection = (focal, focal, width / 2.0, height / 2.0)

        if "vertices" not in raw and mhr_model is None:
            if args.mhr_model is None:
                raise ValueError(f"{raw_path} has no saved vertices; supply --mhr-model")
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

        mask_path = mask_path_for(mask_dir, frame_stem)
        mask = load_mask(mask_path, height, width)
        mask_panel = render_mask_panel(mask)
        masked_image_panel = render_masked_image_panel(image, mask)

        panels = [
            add_label(image, f"image  frame {frame_stem}"),
            add_label(mhr_panel, f"MHR projection fit  frame {frame_stem}"),
        ]
        if args.render_smplx:
            assert smpl_vertices_by_frame is not None and smpl_projection is not None
            frame_index = numeric_frame_id(raw_path)
            smpl_panel = draw_projected_vertices(
                image,
                smpl_vertices_by_frame[frame_index],
                smpl_projection,
                (0, 255, 0),
                args.vertex_stride,
                args.point_radius,
                args.overlay_alpha,
                smpl_distortion,
            )
            panels.append(add_label(smpl_panel, f"SMPL projection fit  frame {frame_stem}"))
        panels.append(add_label(mask_panel, f"mask ({mask_dir.name})  frame {frame_stem}"))
        panels.append(add_label(masked_image_panel, f"masked image  frame {frame_stem}"))
        panels = [scale_panel(panel, args.panel_scale) for panel in panels]
        dashboard = assemble_grid(panels)
        if not cv2.imwrite(str(output_path), dashboard):
            raise RuntimeError(f"Failed to write dashboard: {output_path}")
        written += 1

    print(f"Wrote {written} dashboard images to {args.output_dir}")
    return 0


def main() -> int:
    return render(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

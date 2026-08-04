from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_sapiens_ga_avatar import (
    FrameGeometry,
    expanded_crop_box,
    process_frame,
    project_vertices,
)


def test_projection_ignores_invalid_and_behind_camera_vertices():
    vertices = np.array(
        [[-1.0, -1.0, 2.0], [1.0, 1.0, 2.0], [0.0, 0.0, -1.0], [np.nan, 0.0, 1.0]],
        dtype=np.float32,
    )
    uv = project_vertices(vertices, 100.0, 100.0, 50.0, 50.0)
    assert np.isfinite(uv[:2]).all()
    assert not np.isfinite(uv[2:]).any()
    assert expanded_crop_box(uv, 100, 100, 0.25) == (0, 0, 100, 100)


class _FakeSegmenter:
    def predict(self, crop: np.ndarray) -> np.ndarray:
        mask = np.zeros(crop.shape[:2], dtype=bool)
        h, w = crop.shape[:2]
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return mask


class _FakeDepth:
    def predict(self, crop: np.ndarray) -> np.ndarray:
        return np.full(crop.shape[:2], 3.0, dtype=np.float32)


def test_process_frame_writes_full_resolution_ga_avatar_outputs(tmp_path: Path):
    image_path = tmp_path / "000001.png"
    cv2.imwrite(str(image_path), np.full((100, 120, 3), 127, dtype=np.uint8))
    frame = FrameGeometry(
        "000001",
        image_path,
        np.array(
            [[-0.3, -0.4, 2.0], [0.3, -0.4, 2.0], [0.3, 0.4, 2.0], [-0.3, 0.4, 2.0]],
            dtype=np.float32,
        ),
        100.0,
        100.0,
        60.0,
        50.0,
    )
    mask_path = tmp_path / "masks" / "000001.png"
    depth_path = tmp_path / "sapiens_depth" / "000001.npy"

    box, area = process_frame(
        frame,
        _FakeSegmenter(),
        _FakeDepth(),
        mask_path,
        depth_path,
        margin=0.25,
    )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    depth = np.load(depth_path)
    assert box == (37, 20, 84, 81)
    assert 0.0 < area < 1.0
    assert mask.shape == (100, 120)
    assert depth.shape == (100, 120)
    assert depth.dtype == np.float32
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.all(depth[mask == 0] == 0)
    assert np.all(depth[mask != 0] == 3)

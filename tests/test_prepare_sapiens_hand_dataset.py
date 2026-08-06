import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import prepare_sapiens_hand_dataset as prep


class _FakeSegmenter:
    """Left pixels on the image's left half, right pixels on its right half."""

    def predict(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        h, w = image_bgr.shape[:2]
        left = np.zeros((h, w), dtype=bool)
        right = np.zeros((h, w), dtype=bool)
        left[h // 4 : 3 * h // 4, : w // 2] = True
        right[h // 4 : 3 * h // 4, w // 2 :] = True
        return {"left": left, "right": right}


def _write_source_camera(source_camera: Path, frame_ids: list[str]) -> None:
    images_dir = source_camera / "images"
    masks_dir = source_camera / "masks"
    results_dir = source_camera / "results" / "raw"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    for frame_id in frame_ids:
        cv2.imwrite(str(images_dir / f"{frame_id}.png"), np.full((40, 60, 3), 127, dtype=np.uint8))
        cv2.imwrite(str(masks_dir / f"{frame_id}.png"), np.full((40, 60), 255, dtype=np.uint8))
        (results_dir / f"{frame_id}.npz").write_bytes(b"")


def test_main_writes_split_left_right_masks_and_manifest(tmp_path: Path, monkeypatch) -> None:
    frame_ids = ["000000", "000001"]
    source_camera = tmp_path / "source" / "Camera_B1"
    _write_source_camera(source_camera, frame_ids)

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"valid": frame_ids, "input_root": str(source_camera)}))

    output_root = tmp_path / "output"
    monkeypatch.setattr(
        prep, "SapiensHandSegmenter", lambda *args, **kwargs: _FakeSegmenter()
    )
    monkeypatch.setattr(prep.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sapiens_hand_dataset.py",
            "--split-json", str(split_path),
            "--output-root", str(output_root),
            "--sapiens2-root", str(tmp_path / "unused_sapiens2_root"),
            "--seg-config", str(tmp_path / "unused_config.py"),
            "--seg-checkpoint", str(tmp_path / "unused_checkpoint.pth"),
            "--camera-name", "Camera_B1",
        ],
    )

    assert prep.main() == 0

    camera_out = output_root / "Camera_B1"
    for frame_id in frame_ids:
        combined = cv2.imread(str(camera_out / "hand_masks" / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        left = cv2.imread(str(camera_out / "hand_masks" / "left" / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(camera_out / "hand_masks" / "right" / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        assert left is not None and right is not None and combined is not None
        # Left/right masks are disjoint and their union is exactly the combined mask.
        assert not np.logical_and(left != 0, right != 0).any()
        assert np.array_equal(combined != 0, np.logical_or(left != 0, right != 0))
        assert (left != 0).any()
        assert (right != 0).any()

    manifest = json.loads((output_root / "hand_dataset_manifest.json").read_text())
    assert manifest["hand_class_ids_by_side"] == {"left": 6, "right": 15}
    assert manifest["mask_dirs_by_side"] == {"left": "hand_masks/left", "right": "hand_masks/right"}
    assert {frame["frame_id"] for frame in manifest["frames"]} == set(frame_ids)
    for frame in manifest["frames"]:
        assert frame["left_pixel_fraction"] > 0
        assert frame["right_pixel_fraction"] > 0


def test_main_rejects_overlapping_left_right_masks(tmp_path: Path, monkeypatch) -> None:
    frame_ids = ["000000"]
    source_camera = tmp_path / "source" / "Camera_B1"
    _write_source_camera(source_camera, frame_ids)

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"valid": frame_ids, "input_root": str(source_camera)}))

    class _OverlappingSegmenter:
        def predict(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
            h, w = image_bgr.shape[:2]
            both = np.zeros((h, w), dtype=bool)
            both[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
            return {"left": both, "right": both}

    monkeypatch.setattr(
        prep, "SapiensHandSegmenter", lambda *args, **kwargs: _OverlappingSegmenter()
    )
    monkeypatch.setattr(prep.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sapiens_hand_dataset.py",
            "--split-json", str(split_path),
            "--output-root", str(tmp_path / "output"),
            "--sapiens2-root", str(tmp_path / "unused_sapiens2_root"),
            "--seg-config", str(tmp_path / "unused_config.py"),
            "--seg-checkpoint", str(tmp_path / "unused_checkpoint.pth"),
            "--camera-name", "Camera_B1",
        ],
    )

    with pytest.raises(RuntimeError, match="overlap"):
        prep.main()

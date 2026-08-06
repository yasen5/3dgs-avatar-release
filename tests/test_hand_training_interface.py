from pathlib import Path

import torch

from src.dataset.training_interface import crop_hand_regions, is_hand_dataset


def test_hand_dataset_is_inferred_from_preprocessing_manifest(tmp_path: Path) -> None:
    assert not is_hand_dataset(str(tmp_path))
    (tmp_path / "hand_dataset_manifest.json").write_text("{}")
    assert is_hand_dataset(str(tmp_path))
    assert not is_hand_dataset(str(tmp_path), configured=False)


def test_hand_crops_are_individual_padded_and_synchronized() -> None:
    image = torch.arange(3 * 20 * 30, dtype=torch.float32).reshape(3, 20, 30)
    target = image.clone()
    mask = torch.zeros(1, 20, 30)
    mask[:, 4:8, 2:6] = 1
    mask[:, 10:16, 20:28] = 1

    image_crops, target_crops = crop_hand_regions(
        [image], [target], [mask], padding=0.25
    )

    # The larger hand is 8x6; 25% on each side makes the synchronized crop
    # 12x9 (W x H), and both hands are evaluated in the same dense batch.
    assert image_crops.shape == (2, 3, 9, 12)
    assert torch.equal(image_crops, target_crops)


def test_hand_crops_synchronize_across_views() -> None:
    images = [torch.zeros(3, 24, 32), torch.ones(3, 24, 32)]
    masks = [torch.zeros(1, 24, 32), torch.zeros(1, 24, 32)]
    masks[0][:, 2:6, 2:6] = 1
    masks[1][:, 8:16, 12:20] = 1

    crops, _ = crop_hand_regions(images, images, masks)

    assert crops.shape == (2, 3, 12, 12)
    assert torch.equal(crops[:, 0, 0, 0], torch.tensor([0.0, 1.0]))

from pathlib import Path

import torch

from src.dataset.training_interface import crop_hand_regions, is_hand_dataset


def test_hand_dataset_is_inferred_from_preprocessing_manifest(tmp_path: Path) -> None:
    assert not is_hand_dataset(str(tmp_path))
    (tmp_path / "hand_dataset_manifest.json").write_text("{}")
    assert is_hand_dataset(str(tmp_path))
    assert not is_hand_dataset(str(tmp_path), configured=False)


def test_hand_crops_are_individual_padded_and_synchronized() -> None:
    image = torch.arange(3 * 80 * 120, dtype=torch.float32).reshape(3, 80, 120)
    target = image.clone()
    mask = torch.zeros(1, 80, 120)
    mask[:, 16:32, 8:24] = 1  # 16x16 hand
    mask[:, 40:64, 80:112] = 1  # 24x32 (H x W) hand, the larger one

    image_crops, target_crops = crop_hand_regions(
        [image], [target], [mask], padding=0.25
    )

    # The larger hand is 32x24 (W x H); 25% padding on each side makes the
    # synchronized crop 48x36 (W x H), and both hands are evaluated in the
    # same dense batch.
    assert image_crops.shape == (2, 3, 36, 48)
    assert torch.equal(image_crops, target_crops)


def test_hand_crops_synchronize_across_views() -> None:
    images = [torch.zeros(3, 100, 140), torch.ones(3, 100, 140)]
    masks = [torch.zeros(1, 100, 140), torch.zeros(1, 100, 140)]
    masks[0][:, 10:34, 10:34] = 1  # 24x24 -> padded 36x36
    masks[1][:, 40:80, 60:100] = 1  # 40x40 -> padded 60x60

    crops, _ = crop_hand_regions(images, images, masks)

    assert crops.shape == (2, 3, 60, 60)
    assert torch.equal(crops[:, 0, 0, 0], torch.tensor([0.0, 1.0]))


def test_component_target_selects_single_nearest_hand() -> None:
    # Both hands present in the combined on-disk mask, as in the real
    # single-hand-training case; component_targets picks only the blob
    # nearest the render's own hand position and drops the other one. Boxes
    # are >=32px on a side so the min-crop-size floor is a no-op here and
    # the crop reproduces the source pixels exactly (padding=0).
    image = torch.arange(3 * 80 * 120, dtype=torch.float32).reshape(3, 80, 120)
    mask = torch.zeros(1, 80, 120)
    mask[:, 10:50, 10:50] = 1  # 40x40, centroid ~ (29.5, 29.5)
    mask[:, 20:70, 70:120] = 1  # 50x50, centroid ~ (94.5, 44.5), the larger blob

    crops, _ = crop_hand_regions(
        [image], [image], [mask], padding=0.0, component_targets=[(29.5, 29.5)]
    )

    # Exactly one region selected (the nearer blob), not the larger one.
    assert crops.shape == (1, 3, 40, 40)
    assert torch.allclose(crops[0], image[:, 10:50, 10:50], atol=1e-2)


def test_component_target_ignores_noise_specks_near_target() -> None:
    # A stray 1px segmentation-noise speck sits right next to the target,
    # much closer than either real hand blob -- it must never be selected in
    # place of the real hand: candidates are restricted to the (up to two)
    # largest components before nearest-to-target matching runs.
    image = torch.arange(3 * 80 * 120, dtype=torch.float32).reshape(3, 80, 120)
    mask = torch.zeros(1, 80, 120)
    mask[:, 10:50, 10:50] = 1  # real hand, centroid ~ (29.5, 29.5)
    mask[:, 20:70, 70:120] = 1  # other real hand
    mask[:, 0, 0] = 1  # noise speck, right at the target

    crops, _ = crop_hand_regions(
        [image], [image], [mask], padding=0.0, component_targets=[(0.0, 0.0)]
    )

    assert crops.shape == (1, 3, 40, 40)
    assert torch.allclose(crops[0], image[:, 10:50, 10:50], atol=1e-2)


def test_component_target_none_falls_back_to_both_hands() -> None:
    image = torch.arange(3 * 20 * 30, dtype=torch.float32).reshape(3, 20, 30)
    mask = torch.zeros(1, 20, 30)
    mask[:, 4:8, 2:6] = 1
    mask[:, 10:16, 20:28] = 1

    with_none, _ = crop_hand_regions(
        [image], [image], [mask], padding=0.25, component_targets=[None]
    )
    without_arg, _ = crop_hand_regions([image], [image], [mask], padding=0.25)

    assert torch.equal(with_none, without_arg)


def test_min_crop_size_floors_a_lone_tiny_hand_box() -> None:
    # A single small/fragmented hand blob has no second (larger) box to
    # raise crop_w/crop_h for free the way both-hands mode does, so this
    # would otherwise hand LPIPS's VGG trunk a crop too small for its
    # pooling stack. The floor takes over, clamped here to the source
    # image's own dimensions since it can't exceed them.
    image = torch.arange(3 * 20 * 30, dtype=torch.float32).reshape(3, 20, 30)
    mask = torch.zeros(1, 20, 30)
    mask[:, 4:8, 2:6] = 1  # a tiny 4x4 box, alone

    crops, _ = crop_hand_regions(
        [image], [image], [mask], padding=0.0, component_targets=[(3.5, 5.5)]
    )

    assert crops.shape == (1, 3, 20, 30)

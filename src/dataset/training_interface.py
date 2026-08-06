"""Dataset-facing training-region interface shared by loaders and losses."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


HAND_DATASET_MANIFEST = "hand_dataset_manifest.json"

# Floor on the synchronized crop's output resolution. Both-hands mode is
# self-protecting (a tiny box's crop_w/crop_h gets raised for free by the
# other, usually-larger hand sharing the same batched grid_sample call --
# see crop_hand_regions), but a lone/fragmented single-hand blob (real, from
# imperfect segmentation, or a hand that's mostly out of frame) has no such
# floor and can otherwise produce a crop a few pixels wide, which crashes
# LPIPS's VGG trunk (its 5 stride-2 poolings need >=32px to not collapse to
# a zero-sized feature map). grid_sample resamples continuously regardless
# of the source box's native pixel count, so upsampling a small box to this
# floor is free correctness-wise, not a quality tradeoff.
_MIN_CROP_SIZE = 32


class TrainingDataset(Protocol):
    """The small interface consumed by the upstream training loop."""

    hand_only: bool
    hand_crop_padding: float
    hand_side: str | None


def is_hand_dataset(root_dir: str, configured: bool | None = None) -> bool:
    """Resolve hand mode explicitly, or infer it from preprocessing output."""
    if configured is not None:
        return configured
    return (Path(root_dir) / HAND_DATASET_MANIFEST).is_file()


def _component_boxes(
    mask: torch.Tensor, max_components: int = 2
) -> list[tuple[int, int, int, int]]:
    """Return the ``max_components`` largest hand-component boxes as ``(x1,y1,x2,y2)``.

    The on-disk mask is already restricted to the hand(s) being trained --
    left/right selection is a preprocessing step (see
    ``scripts/prepare_sapiens_hand_dataset.py``), not something this function
    figures out. ``max_components=2`` (both-hands mode) keeps one blob per
    hand; ``max_components=1`` (single-hand mode) keeps only the largest
    blob, so a stray segmentation noise speck can never be selected in place
    of the real hand.
    """
    mask_np = mask.detach().squeeze().cpu().numpy().astype(np.uint8)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected a (1,H,W) hand mask, got {tuple(mask.shape)}")
    count, _, stats_raw, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    stats = np.asarray(stats_raw)
    if count <= 1:
        raise ValueError("Hand-only loss requires at least one ground-truth hand pixel")
    largest = sorted(
        range(1, count), key=lambda i: int(stats[i][cv2.CC_STAT_AREA]), reverse=True
    )[:max_components]
    return [
        (int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
         int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]),
         int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]))
        for s in (stats[i] for i in largest)
    ]


def _padded_box(
    box: tuple[int, int, int, int], padding: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x), min(float(height), y2 + pad_y),
    )


def crop_hand_regions(
    images: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    *,
    padding: float = 0.25,
    max_components: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop hand region(s) and stack them for one synchronized metric call.

    Every connected hand gets 25% padding on each side.  All regions in the
    supplied view batch are then expanded to the same height/width, allowing
    L1, SSIM, LPIPS, and evaluation metrics to run once over a dense batch.

    ``max_components`` applies to every view in the batch: 2 (default) for
    both-hands training, where the on-disk mask has one blob per hand, or 1
    for single-hand training, where the preprocessed mask already contains
    only the trained hand's pixels (see
    ``scripts/prepare_sapiens_hand_dataset.py``).
    """
    if not images or len(images) != len(targets) or len(images) != len(masks):
        raise ValueError("images, targets, and masks must be equally-sized non-empty sequences")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    regions: list[tuple[int, tuple[float, float, float, float]]] = []
    max_w = max_h = 1
    common_hw: tuple[int, int] | None = None
    for index, (image, target, mask) in enumerate(zip(images, targets, masks)):
        if image.shape != target.shape or image.ndim != 3:
            raise ValueError(f"Expected matching (C,H,W) tensors, got {image.shape} and {target.shape}")
        height, width = image.shape[-2:]
        if common_hw is None:
            common_hw = (height, width)
        elif common_hw != (height, width):
            raise ValueError("Synchronized hand crops require equal image dimensions within a batch")
        for raw_box in _component_boxes(mask, max_components):
            box = _padded_box(raw_box, padding, width, height)
            max_w = max(max_w, int(np.ceil(box[2] - box[0])))
            max_h = max(max_h, int(np.ceil(box[3] - box[1])))
            regions.append((index, box))

    assert common_hw is not None
    height, width = common_hw
    crop_w = min(max(max_w, _MIN_CROP_SIZE), width)
    crop_h = min(max(max_h, _MIN_CROP_SIZE), height)
    # Sample every exact padded ROI into the shared largest ROI resolution.
    # This preserves each hand's requested 25% field of view while issuing a
    # single synchronized GPU operation for all hands/views.
    region_indices = torch.tensor(
        [index for index, _ in regions], device=images[0].device, dtype=torch.long
    )
    boxes = torch.tensor(
        [box for _, box in regions], device=images[0].device, dtype=images[0].dtype
    )
    unit_x = torch.linspace(0, 1, crop_w, device=boxes.device, dtype=boxes.dtype)
    unit_y = torch.linspace(0, 1, crop_h, device=boxes.device, dtype=boxes.dtype)
    # x2/y2 use exclusive image coordinates; sample through the final pixel
    # center at x2-1/y2-1.
    xs = boxes[:, 0, None] + unit_x[None] * (boxes[:, 2, None] - boxes[:, 0, None] - 1)
    ys = boxes[:, 1, None] + unit_y[None] * (boxes[:, 3, None] - boxes[:, 1, None] - 1)
    grid_x = xs[:, None, :].expand(-1, crop_h, -1)
    grid_y = ys[:, :, None].expand(-1, -1, crop_w)
    grid = torch.stack(
        (2 * grid_x / max(width - 1, 1) - 1, 2 * grid_y / max(height - 1, 1) - 1),
        dim=-1,
    )
    image_batch = torch.stack(list(images)).index_select(0, region_indices)
    target_batch = torch.stack(list(targets)).index_select(0, region_indices)
    return (
        F.grid_sample(image_batch, grid, mode="bilinear", align_corners=True),
        F.grid_sample(target_batch, grid, mode="bilinear", align_corners=True),
    )

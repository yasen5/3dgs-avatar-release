"""Dataset-facing training-region interface shared by loaders and losses."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


HAND_DATASET_MANIFEST = "hand_dataset_manifest.json"


class TrainingDataset(Protocol):
    """The small interface consumed by the upstream training loop."""

    hand_only: bool
    hand_crop_padding: float


def is_hand_dataset(root_dir: str, configured: bool | None = None) -> bool:
    """Resolve hand mode explicitly, or infer it from preprocessing output."""
    if configured is not None:
        return configured
    return (Path(root_dir) / HAND_DATASET_MANIFEST).is_file()


def _component_boxes(mask: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """Return up to two largest hand-component boxes as ``(x1,y1,x2,y2)``."""
    mask_np = mask.detach().squeeze().cpu().numpy().astype(np.uint8)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected a (1,H,W) hand mask, got {tuple(mask.shape)}")
    count, _, stats_raw, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    stats = np.asarray(stats_raw)
    components = sorted(
        (stats[i] for i in range(1, count)),
        key=lambda stat: int(stat[cv2.CC_STAT_AREA]),
        reverse=True,
    )[:2]
    if not components:
        raise ValueError("Hand-only loss requires at least one ground-truth hand pixel")
    return [
        (int(s[cv2.CC_STAT_LEFT]), int(s[cv2.CC_STAT_TOP]),
         int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]),
         int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]))
        for s in components
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop both hands and stack them for one synchronized metric call.

    Every connected hand gets 25% padding on each side.  All regions in the
    supplied view batch are then expanded to the same height/width, allowing
    L1, SSIM, LPIPS, and evaluation metrics to run once over a dense batch.
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
        for raw_box in _component_boxes(mask):
            box = _padded_box(raw_box, padding, width, height)
            max_w = max(max_w, int(np.ceil(box[2] - box[0])))
            max_h = max(max_h, int(np.ceil(box[3] - box[1])))
            regions.append((index, box))

    assert common_hw is not None
    height, width = common_hw
    crop_w, crop_h = min(max_w, width), min(max_h, height)
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

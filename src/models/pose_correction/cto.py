"""GA-Avatar's Canonical Template Optimization (CTO), wired through the same
`pose_correction` config-group slot the MHR pipeline uses for its own
photometric-loss-driven body-model parameter optimization (DirectPoseOptimization).

Unlike DirectPoseOptimization, CTO never touches per-frame pose theta -- the
paper holds pose fixed throughout training and only optimizes the body-model
level {beta, joint_offset, face_offset} (see SMPLXBodyModel.cto_parameters).
This module's whole job is to (a) make `metadata['body_model']` a real
submodule of GaussianConverter, so `.cuda()`/the optimizer/grad-clipping all
reach its CTO parameters, and (b) supply L_joint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.body_models.metadata import ModelMetadata
from src.scene.cameras import Camera


class SMPLXTemplateOptimization(nn.Module):
    def __init__(self, cfg: DictConfig, metadata: ModelMetadata) -> None:
        super().__init__()
        self.cfg = cfg
        self.body_model = metadata["body_model"]  # nn.Module submodule: beta/joint_offset/face_offset

        joint_offset_gt = metadata.get("joint_offset_gt", None)
        if joint_offset_gt is None:
            joint_offset_gt = torch.zeros_like(self.body_model.joint_offset)
        self.register_buffer("joint_offset_gt", joint_offset_gt.clone())

    def forward(self, camera: Camera, iteration: int) -> tuple[Camera, dict[str, torch.Tensor]]:
        loss_joint = nn.functional.mse_loss(self.body_model.joint_offset, self.joint_offset_gt)
        return camera, {"joint": loss_joint}

    def regularization(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}


def get_pose_correction_extra() -> dict[str, type[nn.Module]]:
    """Extra `pose_correction` factory entries contributed by this module --
    kept separate from src/models/pose_correction/pose_correction.py's own
    `get_pose_correction` dict so that module doesn't need a `smplx`/
    BodyModel import (it stays MHR-only)."""
    return {"smplx_cto": SMPLXTemplateOptimization}

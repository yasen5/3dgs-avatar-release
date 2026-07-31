from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.dataset.mhr_native import MHRMetadata
from src.models.deformer.non_rigid import get_non_rigid_deform
from src.models.deformer.rigid import get_rigid_deform
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel


class Deformer(nn.Module):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__()
        self.cfg = cfg
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)

    def forward(
        self, gaussians: GaussianModel, camera: Camera, iteration: int, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        loss_reg: dict[str, torch.Tensor] = {}
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

def get_deformer(cfg: DictConfig, metadata: MHRMetadata) -> Deformer:
    return Deformer(cfg, metadata)

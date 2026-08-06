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

        if cfg.rigid.get('disable_grad', False):
            self.rigid.requires_grad_(False)
        if cfg.non_rigid.get('disable_grad', False):
            self.non_rigid.requires_grad_(False)

    def forward(
        self, gaussians: GaussianModel, camera: Camera, iteration: int, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        loss_reg: dict[str, torch.Tensor] = {}
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

    def forward_batch(
        self, gaussians: GaussianModel, cameras: list[Camera], iteration: int, compute_loss: bool = True
    ) -> tuple[list[GaussianModel], dict[str, torch.Tensor]]:
        """Batched equivalent of forward(): uses non_rigid/rigid's own
        forward_batch when both implement one (currently HashGridwithMLP and
        SkinningField -- this repo's paper-default config), otherwise falls
        back to calling forward() once per camera so every other
        rigid/non-rigid combination (VertexLBS, GAAvatarGeo, MLP, HannwMLP)
        keeps its exact previous per-view behavior unchanged."""
        if not (hasattr(self.non_rigid, 'forward_batch') and hasattr(self.rigid, 'forward_batch')):
            out: list[GaussianModel] = []
            loss_accum: dict[str, list[torch.Tensor]] = {}
            for cam in cameras:
                dg, loss_reg = self.forward(gaussians, cam, iteration, compute_loss)
                out.append(dg)
                for k, v in loss_reg.items():
                    loss_accum.setdefault(k, []).append(v)
            loss_reg_out = {k: torch.stack(v).mean() for k, v in loss_accum.items()}
            return out, loss_reg_out

        gaussians_list, loss_non_rigid = self.non_rigid.forward_batch(gaussians, iteration, cameras, compute_loss)
        gaussians_list = self.rigid.forward_batch(gaussians_list, iteration, cameras)
        return gaussians_list, loss_non_rigid

def get_deformer(cfg: DictConfig, metadata: MHRMetadata) -> Deformer:
    return Deformer(cfg, metadata)

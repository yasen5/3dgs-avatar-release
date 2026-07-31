from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.constants import ADAM_EPS
from src.dataset.mhr_native import MHRMetadata
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel

from .deformer import get_deformer
from .pose_correction import get_pose_correction
from .texture import get_texture


class GaussianConverter(nn.Module):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

        self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)
        self.deformer = get_deformer(cfg.model.deformer, metadata)
        self.texture = get_texture(cfg.model.texture, metadata)

        if cfg.model.pose_correction.get('disable_grad', False):
            self.pose_correction.requires_grad_(False)
        if cfg.model.texture.get('disable_grad', False):
            self.texture.requires_grad_(False)

        self.set_optimizer()

    def set_optimizer(self) -> None:
        opt_params = [
            {'params': [p for p in self.deformer.rigid.parameters() if p.requires_grad],
             'lr': self.cfg.opt.rigid_lr},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n and p.requires_grad],
             'lr': self.cfg.opt.non_rigid_lr},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n and p.requires_grad],
             'lr': self.cfg.opt.nr_latent_lr, 'weight_decay': self.cfg.opt.latent_weight_decay},
            {'params': [p for p in self.pose_correction.parameters() if p.requires_grad],
             'lr': self.cfg.opt.pose_correction_lr},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n and p.requires_grad],
             'lr': self.cfg.opt.texture_lr},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n and p.requires_grad],
             'lr': self.cfg.opt.tex_latent_lr, 'weight_decay': self.cfg.opt.latent_weight_decay},
        ]
        # every param group above sets its own 'lr', so Adam's top-level lr is never used
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0., eps=ADAM_EPS)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(
        self, gaussians: GaussianModel, camera: Camera, iteration: int, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor], torch.Tensor]:
        loss_reg: dict[str, torch.Tensor] = {}
        # loss_reg.update(gaussians.get_opacity_loss())
        camera, loss_reg_pose = self.pose_correction(camera, iteration)

        # pose augmentation
        pose_noise = self.cfg.pipeline.pose_noise
        if self.training and pose_noise > 0 and np.random.uniform() <= self.cfg.pipeline.pose_noise_apply_prob:
            camera = camera.copy()
            camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise

        deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss)

        loss_reg.update(loss_reg_pose)
        loss_reg.update(loss_reg_deformer)

        color_precompute = self.texture(deformed_gaussians, camera)

        return deformed_gaussians, loss_reg, color_precompute

    def optimize(self) -> None:
        grad_clip = self.cfg.opt.grad_clip
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

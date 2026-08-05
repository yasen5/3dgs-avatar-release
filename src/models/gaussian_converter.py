from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.constants import ADAM_EPS
from src.dataset.mhr_native import MHRMetadata
from src.models.network_utils import TriplaneEncoder
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

        # GA-Avatar: one Triplane feature encoder shared between GeoNet
        # (deformer.non_rigid) and RGBNet (texture), matching the paper's
        # single F_tri feeding both branches. Only built when the config
        # declares a `model.triplane` block (GA-Avatar presets); absent for
        # MHR configs, which don't use a triplane at all.
        triplane_cfg = cfg.model.get('triplane', None)
        if triplane_cfg is not None:
            self.triplane = TriplaneEncoder(
                triplane_cfg.body, triplane_cfg.face, metadata['face_vertex_mask']
            )
            metadata['triplane'] = self.triplane

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
            {'params': [p for p in self.triplane.parameters() if p.requires_grad] if hasattr(self, 'triplane') else [],
             'lr': self.cfg.opt.get('triplane_lr', self.cfg.opt.non_rigid_lr)},
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

        lr_schedule = self.cfg.opt.get('lr_schedule', 'exponential')
        if lr_schedule == 'exponential':
            gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
            self.scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif lr_schedule == 'milestone':
            # GA-Avatar's paper schedule: lr -> 10% at 75% of iterations,
            # -> 1% (of the ORIGINAL lr, i.e. another x0.1) at 95% -- two
            # discrete x0.1 drops, unlike the per-step exponential decay
            # every other mode here uses.
            iterations = self.cfg.opt.iterations
            milestones = [int(0.75 * iterations), int(0.95 * iterations)]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        else:
            raise ValueError(f"Unknown opt.lr_schedule: {lr_schedule!r}")

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

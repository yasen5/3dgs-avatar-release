from __future__ import annotations

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

        deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss)

        loss_reg.update(loss_reg_pose)
        loss_reg.update(loss_reg_deformer)

        color_precompute = self.texture(deformed_gaussians, camera)

        return deformed_gaussians, loss_reg, color_precompute

    def forward_batch(
        self, gaussians: GaussianModel, cameras: list[Camera], iteration: int, compute_loss: bool = True
    ) -> tuple[list[GaussianModel], dict[str, torch.Tensor], torch.Tensor]:
        """Batched equivalent of forward(): every view's pose-correction +
        deformer + texture forward pass runs as a handful of batched tensor
        ops instead of one Python-level call per view -- see forward_batch on
        DirectPoseOptimization/Deformer/ColorMLP for what's actually batched
        underneath. Falls back to forward()'s original per-view loop when the
        configured pose_correction/texture don't implement a batched path
        (CTO pose correction, SH2RGB/GAAvatarRGB textures), so every other
        config keeps its exact previous behavior.

        Unlike forward()'s per-view loss_reg dict, this returns ONE dict of
        values already averaged over the batch -- train.py's per-render_pkg
        loss_reg_sums[name] += value, then /= n_views, recovers the exact
        same number when every render_pkg is handed the same pre-averaged
        value (summing the same value B times then dividing by B is a
        no-op), so render_batch can hand this single dict to every view's
        RenderOutput unchanged.
        """
        if not hasattr(self.pose_correction, 'forward_batch'):
            pcs, loss_regs, colors_list = [], [], []
            for camera in cameras:
                pc, loss_reg, colors_precomp = self.forward(gaussians, camera, iteration, compute_loss)
                pcs.append(pc)
                loss_regs.append(loss_reg)
                colors_list.append(colors_precomp)
            loss_accum: dict[str, list[torch.Tensor]] = {}
            for loss_reg in loss_regs:
                for k, v in loss_reg.items():
                    loss_accum.setdefault(k, []).append(v)
            loss_reg_out = {k: torch.stack(v).mean() for k, v in loss_accum.items()}
            return pcs, loss_reg_out, torch.stack(colors_list)

        cams_updated, loss_reg_pose = self.pose_correction.forward_batch(cameras, iteration)
        gaussians_list, loss_reg_deformer = self.deformer.forward_batch(gaussians, cams_updated, iteration, compute_loss)
        loss_reg = {**loss_reg_pose, **loss_reg_deformer}

        if hasattr(self.texture, 'forward_batch'):
            colors = self.texture.forward_batch(gaussians_list, cams_updated)
        else:
            colors = torch.stack([self.texture(g, cam) for g, cam in zip(gaussians_list, cams_updated)])

        return gaussians_list, loss_reg, colors

    def optimize(self) -> None:
        grad_clip = self.cfg.opt.grad_clip
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

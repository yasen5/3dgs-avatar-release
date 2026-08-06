#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import os
from typing import cast

import torch
from omegaconf import DictConfig

from src.dataset import load_dataset
from src.dataset.mhr_native import MHRMetadata, MHRNativeDataset
from src.models import GaussianConverter
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel


class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg: DictConfig, gaussians: GaussianModel, save_dir: str) -> None:
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset: MHRNativeDataset = load_dataset(cfg.dataset, split='train')
        # get_metadata() always populates the full MHRMetadata (not just the
        # split-independent MHRCanonicalMetadata subset) for split='train'.
        self.metadata = cast(MHRMetadata, self.train_dataset.metadata)
        self.test_dataset: MHRNativeDataset
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent: float = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self) -> None:
        self.converter.train()

    def eval(self) -> None:
        self.converter.eval()

    def optimize(self, iteration: int) -> None:
        gaussians_delay = self.cfg.model.gaussian.delay
        if iteration >= gaussians_delay:
            assert self.gaussians.optimizer is not None
            self.gaussians.optimizer.step()
        assert self.gaussians.optimizer is not None
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

    def convert_gaussians(
        self, viewpoint_camera: Camera, iteration: int, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor], torch.Tensor]:
        result: tuple[GaussianModel, dict[str, torch.Tensor], torch.Tensor] = self.converter(
            self.gaussians, viewpoint_camera, iteration, compute_loss
        )
        return result

    def convert_gaussians_batch(
        self, cameras: list[Camera], iteration: int, compute_loss: bool = True
    ) -> tuple[list[GaussianModel], dict[str, torch.Tensor], torch.Tensor]:
        return self.converter.forward_batch(self.gaussians, cameras, iteration, compute_loss)

    def get_skinning_loss(self) -> torch.Tensor:
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        colors = self.bake_colors(iteration)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), colors=colors)

    def bake_colors(self, iteration: int, ref_view: Camera | None = None) -> torch.Tensor:
        """Run the (pose/view-dependent) texture MLP for a single reference
        view and freeze its output as static per-Gaussian RGB, since a PLY
        has no notion of the pose- and view-dependent appearance this model
        actually produces.
        """
        if ref_view is None:
            ref_view = self.test_dataset[0]
        was_training = self.converter.training
        self.eval()
        with torch.no_grad():
            _, _, colors = self.convert_gaussians(ref_view, iteration, compute_loss=False)
        if was_training:
            self.train()
        return colors

    def save_checkpoint(self, iteration: int) -> None:
        # Write to a temp file and only os.replace() it onto the final name
        # once the write has fully succeeded. A checkpoint that dies partway
        # through (e.g. disk full) then leaves the *previous* checkpoint (if
        # any) intact instead of clobbering it with a truncated, unloadable
        # file -- and doesn't propagate, since a save failure this late in
        # training must not throw away an otherwise-complete run.
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        final_path = self.save_dir + "/ckpt" + str(iteration) + ".pth"
        tmp_path = final_path + ".tmp"
        try:
            torch.save((self.gaussians.capture(),
                        self.converter.state_dict(),
                        self.converter.optimizer.state_dict(),
                        self.converter.scheduler.state_dict(),
                        iteration), tmp_path)
            os.replace(tmp_path, final_path)
        except Exception as e:
            print(f"[ITER {iteration}] WARNING: checkpoint save failed ({e!r}); continuing without it")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_checkpoint(self, path: str) -> None:
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path, weights_only=False)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)

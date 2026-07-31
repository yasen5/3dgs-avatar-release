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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.scene import GaussianModel, Scene


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    color_ref_view = config.get('color_ref_view', None)

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        # Appearance here is produced by a per-frame/per-view neural texture
        # network (models/texture/texture.py), not a static per-point color
        # -- so there is no single "true" color to store. We bake one
        # representative snapshot (from color_ref_view, or the first test
        # view by default) as standard red/green/blue PLY fields purely for
        # visualization in generic viewers; the exact latent features are
        # saved alongside it so the real model can still be reconstructed
        # exactly.
        view = None
        if color_ref_view is not None:
            for cand_idx in range(len(scene.test_dataset)):
                cand = scene.test_dataset[cand_idx]
                if cand.image_name == color_ref_view:
                    view = cand
                    break
            if view is None:
                raise ValueError(f"color_ref_view '{color_ref_view}' not found in test_dataset")
        colors = scene.bake_colors(config.opt.iterations, ref_view=view)

        point_cloud_path = os.path.join(scene.save_dir, "point_cloud/iteration_{}".format(config.opt.iterations))
        ply_path = os.path.join(point_cloud_path, "point_cloud.ply")
        gaussians.save_ply(ply_path, colors=colors)
        print("Saved point cloud to", ply_path)


if __name__ == "__main__":
    main()

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

# Renders the test set from a canonical Gaussian .ply (geometry/latent
# features) plus the converter (deformer/texture/pose-correction) weights
# taken separately from a training checkpoint -- instead of loading
# everything from the single .pth via scene.load_checkpoint(). Used to
# verify that a ply + converter-weights reload reproduces byte-identical
# renders to loading the full .pth checkpoint directly.

from __future__ import annotations

import os
import sys
from os import makedirs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from src.gaussian_renderer import render
from src.scene import GaussianModel, Scene


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()

        ply_path = config.ply_path
        ckpt_path = config.load_ckpt

        gaussians.load_ply_generic(ply_path)
        _, converter_sd, _, _, _ = torch.load(ckpt_path, weights_only=False)
        scene.converter.load_state_dict(converter_sd)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        for idx in trange(len(scene.test_dataset), desc="Rendering from ply"):
            view = scene.test_dataset[idx]
            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                 compute_loss=False, return_opacity=False)
            rendering = render_pkg["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

        print("Done rendering from ply to", render_path)


if __name__ == "__main__":
    main()

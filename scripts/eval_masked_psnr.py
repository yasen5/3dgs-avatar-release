"""Render a checkpoint's test-view set and report both full-frame PSNR (matches
scripts/render.py's metric, comparable to the 3DGA paper's published full-scene
numbers) and person-silhouette-masked PSNR (background pixels excluded via
camera.original_mask), for A/B comparison against a baseline run."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.gaussian_renderer import render
from src.scene import GaussianModel, Scene


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(pred)
    se = (pred - gt) ** 2 * mask
    mse = se.sum() / mask.sum().clamp_min(1.0)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or f'./exp/{config.name}'
    config.mode = 'test'

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = f"{scene.save_dir}/ckpt{config.opt.iterations}.pth"
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        full_psnrs = []
        masked_psnrs = []
        for idx in range(len(scene.test_dataset)):
            view = scene.test_dataset[idx]
            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                 compute_loss=False, return_opacity=False)
            rendering = render_pkg["render"].clamp(0.0, 1.0)
            gt = view.original_image[:3, :, :]
            mask = view.original_mask

            mse_full = ((rendering - gt) ** 2).mean()
            full_psnrs.append((10 * torch.log10(1.0 / mse_full.clamp_min(1e-10))).item())
            masked_psnrs.append(masked_psnr(rendering, gt, mask).item())

        full_psnr = float(np.mean(full_psnrs))
        person_psnr = float(np.mean(masked_psnrs))
        print(f"n_test_views={len(scene.test_dataset)}")
        print(f"full_frame_psnr={full_psnr:.4f}")
        print(f"person_silhouette_psnr={person_psnr:.4f}")
        np.savez(f"{config.exp_dir}/masked_psnr_results.npz",
                 full_frame_psnr=full_psnr, person_silhouette_psnr=person_psnr,
                 n_test_views=len(scene.test_dataset))


if __name__ == "__main__":
    main()

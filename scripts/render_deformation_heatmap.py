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

"""Colorize the canonical point cloud of a trained (pure-deformation) run by
how much each Gaussian moves across the whole training sequence.

For every training frame, the deformer (pose_correction -> rigid ->
non-rigid, i.e. the same path used at render time) is applied to the frozen
canonical Gaussians and the per-point displacement norm from the canonical
position is recorded. The per-point "total deformation" is the sum of those
displacement norms over all frames -- a proxy for how much each point on the
body actually moves (breathing, cloth, etc.) over the course of the video.
That scalar field is mapped through a matplotlib colormap and written out as
a colored PLY at the canonical (undeformed) positions, alongside the raw
per-point stats (sum/mean/max) in a .npz for further analysis.

Meant to be paired with a "pure deformation" training run, e.g.
configs/option/mask_only.yaml, where the canonical Gaussians and appearance
networks are frozen and only the rigid + non-rigid deformer can move -- so
the displacement measured here isn't confounded by a moving canonical shape.

Usage:
    python scripts/render_deformation_heatmap.py \
        dataset=chestvid_dense_mhr \
        option=mask_only \
        exp_dir=/absolute/path/to/exp/chestvid_deform_only \
        load_ckpt=/absolute/path/to/exp/chestvid_deform_only/ckpt3000.pth \
        opt.iterations=3000 \
        wandb_disable=true
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from src.scene import GaussianModel, Scene


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    colormap_name = config.get('heatmap_colormap', 'inferno')
    pct_clip = config.get('heatmap_pct_clip', 99.0)
    stat = config.get('heatmap_stat', 'sum')
    if stat not in ('sum', 'mean', 'max'):
        raise ValueError(f"heatmap_stat must be one of sum/mean/max, got {stat!r}")

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        canonical_xyz = gaussians.get_xyz.detach().clone()
        n_points = canonical_xyz.shape[0]
        n_frames = len(scene.train_dataset)

        sum_disp = torch.zeros(n_points, device=canonical_xyz.device)
        max_disp = torch.zeros(n_points, device=canonical_xyz.device)

        for idx in trange(n_frames, desc="Accumulating per-frame deformation"):
            view = scene.train_dataset[idx]
            deformed_gaussians, _, _ = scene.convert_gaussians(view, config.opt.iterations, compute_loss=False)
            disp = torch.norm(deformed_gaussians.get_xyz - canonical_xyz, p=2, dim=1)
            sum_disp += disp
            max_disp = torch.maximum(max_disp, disp)

        mean_disp = sum_disp / max(n_frames, 1)

        stat_tensor = {'sum': sum_disp, 'mean': mean_disp, 'max': max_disp}[stat]
        values = stat_tensor.cpu().numpy()

        print(f"points: {n_points}, frames: {n_frames}")
        print(f"per-point total (summed) displacement: mean={sum_disp.mean().item():.6f} "
              f"max={sum_disp.max().item():.6f}")
        print(f"per-point mean-per-frame displacement:  mean={mean_disp.mean().item():.6f} "
              f"max={mean_disp.max().item():.6f}")
        print(f"per-point max-over-frames displacement: mean={max_disp.mean().item():.6f} "
              f"max={max_disp.max().item():.6f}")
        print(f"coloring PLY by '{stat}' displacement, clipped at the {pct_clip}th percentile")

        vmin = np.percentile(values, 100. - pct_clip)
        vmax = np.percentile(values, pct_clip)
        vmax = max(vmax, vmin + 1e-12)
        normed = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        cmap = matplotlib.colormaps[colormap_name]
        colors = cmap(normed)[:, :3]  # drop alpha
        colors_t = torch.from_numpy(colors).float().to(canonical_xyz.device)

        point_cloud_path = os.path.join(scene.save_dir, "point_cloud/iteration_{}".format(config.opt.iterations))
        os.makedirs(point_cloud_path, exist_ok=True)
        ply_path = os.path.join(point_cloud_path, "deformation_heatmap.ply")

        # write the heatmap PLY at the canonical (undeformed) positions
        canonical_gaussians = gaussians.clone()
        canonical_gaussians._xyz = canonical_xyz
        canonical_gaussians.save_ply(ply_path, colors=colors_t)
        print("saved heatmap PLY to", ply_path)

        stats_path = os.path.join(point_cloud_path, "deformation_stats.npz")
        np.savez(
            stats_path,
            canonical_xyz=canonical_xyz.cpu().numpy(),
            sum_displacement=sum_disp.cpu().numpy(),
            mean_displacement=mean_disp.cpu().numpy(),
            max_displacement=max_disp.cpu().numpy(),
            n_frames=n_frames,
        )
        print("saved per-point stats to", stats_path)


if __name__ == "__main__":
    main()

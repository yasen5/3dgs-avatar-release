"""Recolor an existing deformation_heatmap.ply from its companion
deformation_stats.npz, using a [vmin, vmax] percentile-stretched colormap
instead of a [0, vmax] one (the latter crushes a shared-baseline-dominated
signal into a thin slice of the colormap). Avoids rerunning the deformer
over every training frame.

Usage:
    python scripts/recolor_deformation_heatmap.py <exp_dir_point_cloud_iter_dir> \
        [--stat sum|mean|max] [--colormap inferno] [--pct-clip 99.0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np
import torch

from src.scene.gaussian_model import GaussianModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud_dir")
    parser.add_argument("--stat", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--colormap", default="inferno")
    parser.add_argument("--pct-clip", type=float, default=99.0)
    args = parser.parse_args()

    ply_path = os.path.join(args.point_cloud_dir, "deformation_heatmap.ply")
    stats_path = os.path.join(args.point_cloud_dir, "deformation_stats.npz")

    stats = np.load(stats_path)
    values = stats[{"sum": "sum_displacement", "mean": "mean_displacement", "max": "max_displacement"}[args.stat]]

    vmin = np.percentile(values, 100. - args.pct_clip)
    vmax = np.percentile(values, args.pct_clip)
    vmax = max(vmax, vmin + 1e-12)
    normed = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    cmap = matplotlib.colormaps[args.colormap]
    colors = cmap(normed)[:, :3]
    colors_t = torch.from_numpy(colors).float().cuda()

    gaussians = GaussianModel.__new__(GaussianModel)
    gaussians.load_ply_generic(ply_path)
    gaussians.save_ply(ply_path, colors=colors_t)
    print(f"recolored {ply_path} using '{args.stat}' displacement, "
          f"range [{vmin:.3f}, {vmax:.3f}] (pct_clip={args.pct_clip})")


if __name__ == "__main__":
    main()

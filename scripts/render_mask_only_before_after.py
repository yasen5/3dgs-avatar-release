"""Ablation visualization: compare the deformed (posed) Gaussian point cloud
before vs. after training with only the mask loss active (see
configs/option/mask_only.yaml). The canonical Gaussians and appearance/pose
networks are frozen in that ablation, so any difference in the deformed point
cloud comes purely from the rigid + non-rigid deformer networks updating on
the mask loss.

"Before" = the freshly-initialized model (deformer weights at init).
"After"  = the model loaded from the ablation run's final checkpoint.

Usage:
    python scripts/render_mask_only_before_after.py \
        exp_dir=exp/mask_only_1k ckpt=exp/mask_only_1k/ckpt1000.pth
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf

from src.gaussian_renderer import render
from src.scene import GaussianModel, Scene

EXP_DIR = "exp/mask_only_1k"
CKPT = "exp/mask_only_1k/ckpt1000.pth"
OUT_PNG = "/tmp/claude-1000/-home-yasen-lesegment-train-third-party-texture-appearance-3dgs-avatar-release/35291b52-5cab-4a6a-864f-9cf8130d3ce9/scratchpad/mask_only_before_after.png"
VIEW_IDX = 0

with initialize_config_dir(version_base=None, config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="config", overrides=[
        "dataset=trimmed_final_mhr",
        "option=mask_only",
        "name=mask_only_1k",
        "opt.iterations=1000",
    ])
OmegaConf.set_struct(cfg, False)
cfg.exp_dir = EXP_DIR

with torch.no_grad():
    # "before": freshly initialized model, no training steps taken
    gaussians_before = GaussianModel(cfg.model.gaussian)
    scene_before = Scene(cfg, gaussians_before, cfg.exp_dir)
    scene_before.eval()

    bg_color = [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    view = scene_before.train_dataset[VIEW_IDX]

    render_before = render(view, 0, scene_before, cfg.pipeline, background,
                            compute_loss=False, return_opacity=True)
    xyz_before = render_before["deformed_gaussian"].get_xyz.detach().cpu().numpy()
    mask_before = render_before["opacity_render"]
    assert mask_before is not None
    mask_before_np = mask_before.squeeze(0).detach().cpu().numpy()

    # "after": same architecture, weights loaded from the ablation checkpoint
    gaussians_after = GaussianModel(cfg.model.gaussian)
    scene_after = Scene(cfg, gaussians_after, cfg.exp_dir)
    scene_after.eval()
    scene_after.load_checkpoint(CKPT)

    render_after = render(view, cfg.opt.iterations, scene_after, cfg.pipeline, background,
                           compute_loss=False, return_opacity=True)
    xyz_after = render_after["deformed_gaussian"].get_xyz.detach().cpu().numpy()
    mask_after = render_after["opacity_render"]
    assert mask_after is not None
    mask_after_np = mask_after.squeeze(0).detach().cpu().numpy()

displacement = np.linalg.norm(xyz_after - xyz_before, axis=1)
print("view:", view.image_name)
print("num points:", xyz_before.shape[0])
print("mean displacement:", displacement.mean(), "max displacement:", displacement.max())

# axes: (x=width, y=depth, z=height), matching render_cano_pose.py's convention
pts_before = xyz_before[:, [0, 2, 1]]
pts_after = xyz_after[:, [0, 2, 1]]

all_pts = np.concatenate([pts_before, pts_after], axis=0)
mins, maxs = all_pts.min(0), all_pts.max(0)
ranges = maxs - mins

fig = plt.figure(figsize=(16, 8))

ax1 = cast(Axes3D, fig.add_subplot(141, projection="3d"))
ax1.scatter(*pts_before.T, s=0.4, c=pts_before[:, 2], cmap="viridis")
ax1.set_title("before (init deformer)")

ax2 = cast(Axes3D, fig.add_subplot(142, projection="3d"))
ax2.scatter(*pts_after.T, s=0.4, c=pts_after[:, 2], cmap="viridis")
ax2.set_title("after 1k iters (mask loss only)")

ax3 = cast(Axes3D, fig.add_subplot(143, projection="3d"))
sc = ax3.scatter(*pts_after.T, s=0.4, c=displacement, cmap="inferno")
ax3.set_title("per-point displacement")
fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.1)

for ax in (ax1, ax2, ax3):
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(ranges)
    ax.view_init(elev=8, azim=100)
    ax.invert_zaxis()
    ax.set_axis_off()

ax4 = fig.add_subplot(144)
h, w = mask_before_np.shape
combined = np.zeros((h, w, 3))
combined[..., 0] = mask_before_np  # red = before
combined[..., 1] = mask_after_np   # green = after
ax4.imshow(combined)
ax4.set_title("rendered mask: before=red, after=green")
ax4.set_axis_off()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print("wrote", OUT_PNG)

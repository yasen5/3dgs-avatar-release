"""Render CoreView_377's canonical ('big') pose: point cloud + shaded mesh, side by side."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
from src.dataset.mhr_native import MHRNativeDataset  # noqa: E402

with initialize_config_dir(version_base=None, config_dir=f"{REPO}/configs"):
    cfg = compose(config_name="config", overrides=["dataset=zju_377_mhr"])

ds = MHRNativeDataset(cfg.dataset, split="train")
verts = ds.metadata["cano_verts"][:, [0, 2, 1]]  # (x=width, y=depth, z=height)
faces = ds.metadata["faces"]

ranges = verts.max(0) - verts.min(0)
fig = plt.figure(figsize=(12, 7))

# mpl_toolkits stubs don't overload Figure.add_subplot on projection="3d",
# so it comes back typed as the base Axes -- cast to the real Axes3D.
ax1 = cast(Axes3D, fig.add_subplot(121, projection="3d"))
ax1.scatter(*verts.T, s=0.4, c=verts[:, 2], cmap="viridis")
ax1.set_title("canonical point cloud")

ax2 = cast(Axes3D, fig.add_subplot(122, projection="3d"))
ax2.add_collection3d(Poly3DCollection(
    verts[faces], facecolor=(0.75, 0.75, 0.9), edgecolor="k", linewidths=0.03, alpha=0.95))
ax2.set_title("canonical mesh")

for ax in (ax1, ax2):
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.set_box_aspect(ranges)
    ax.view_init(elev=8, azim=100)
    ax.invert_zaxis()
    ax.set_axis_off()

out = "/tmp/claude-1000/-home-yasen-lesegment-train-third-party-texture-appearance-3dgs-avatar-release/8b27ac66-029f-42ff-a9fc-cefdf7eba389/scratchpad/cano_pose_render.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

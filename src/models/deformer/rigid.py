from __future__ import annotations

import igl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch3d.ops import knn_points

from src.dataset.mhr_native import MHRMetadata
from src.models.network_utils import get_skinning_mlp
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel
from src.utils.general_utils import build_rotation


class RigidDeform(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians: GaussianModel, iteration: int, camera: Camera) -> GaussianModel:
        raise NotImplementedError

    def regularization(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

def create_voxel_grid(d: int, h: int, w: int, device: str = 'cpu') -> torch.Tensor:
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

class SkinningField(RigidDeform):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        self.cano_verts = metadata["cano_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        self.aabb = metadata["aabb"]
        self.faces = metadata["faces"]
        self.cano_mesh = metadata["cano_mesh"]

        self.distill: bool = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        self.n_joints = cfg.d_out
        self.grid: torch.Tensor
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        self.lbs_network = get_skinning_mlp(3, cfg.d_out, cfg.skinning_network)
        self.lbs_voxel_final: torch.Tensor

        self.cano_verts_norm: torch.Tensor
        self.skinning_weights_t: torch.Tensor
        cano_verts_t = torch.from_numpy(self.cano_verts).float()
        self.register_buffer("cano_verts_norm", self.aabb.normalize(cano_verts_t, sym=True))
        self.register_buffer("skinning_weights_t", torch.from_numpy(self.skinning_weights).float())


    def precompute(self, recompute_skinning: bool = True) -> None:
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, self.n_joints, d, h, w)

    def predict_skinning_weights(self, xyz_norm: torch.Tensor) -> torch.Tensor:
        """3DGA-style skinning weights: an MLP predicts a logit-space offset on
        top of a default skinning weight looked up from the nearest canonical
        mesh vertex (Eq. 7 in 3dga_paper.txt), instead of predicting the full
        weight distribution from scratch. The nearest-neighbor lookup is
        non-differentiable; only the offset MLP receives gradient."""
        with torch.no_grad():
            _, idx, _ = knn_points(xyz_norm[None], self.cano_verts_norm[None], K=1)
        default_w = self.skinning_weights_t[idx.view(-1)]
        offset_logits = self.lbs_network(xyz_norm)
        pts_W = self.softmax(torch.log(default_w + 1e-9) + offset_logits)
        return pts_W

    def get_forward_transform(self, xyz: torch.Tensor, tfs: torch.Tensor) -> torch.Tensor:
        if self.distill:
            self.precompute(recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            pts_W = self.predict_skinning_weights(xyz)
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd

    def sample_skinning_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.cano_verts[self.faces[face_idx, 0], :],
            self.cano_verts[self.faces[face_idx, 1], :],
            self.cano_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

        points_skinning_t = torch.from_numpy(points_skinning).cuda()
        pts_W_t = torch.from_numpy(pts_W).cuda()
        return points_skinning_t, pts_W_t

    def softmax(self, logit: torch.Tensor) -> torch.Tensor:
        return F.softmax(logit, dim=-1)

    def get_skinning_loss(self) -> torch.Tensor:
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.aabb.normalize(pts_skinning, sym=True)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(self.n_joints, -1).permute(1, 0)
        else:
            pred_weights = self.predict_skinning_weights(pts_skinning)
        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # breakpoint()

        return skinning_loss


    def forward(self, gaussians: GaussianModel, iteration: int, camera: Camera) -> GaussianModel:
        tfs = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        T_fwd = self.get_forward_transform(xyz_norm, tfs)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        deformed_gaussians.rotation_precomp = rotation_bar
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self) -> dict[str, torch.Tensor]:
        loss_skinning = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning
        }

class VertexLBS(RigidDeform):
    """GA-Avatar's rigid deformer: one Gaussian per canonical mesh vertex, so
    each Gaussian's skinning weight is simply that vertex's own (subdivision-
    propagated) row of the body model's dense skinning weights -- no learned
    skinning field, no nearest-neighbor lookup (contrast with SkinningField
    above, which is needed only when Gaussians are NOT 1:1 with mesh
    vertices).

    Per-frame bone transforms are recomputed live each forward() call from
    `metadata['body_model']` (whose CTO parameters -- beta, joint_offset --
    are being optimized) composed with the frame's fixed pose
    (`camera.rots`) and world-alignment (`camera.align_matrix`), rather than
    reusing `camera.bone_transforms`'s dataset-init-time snapshot, so that
    gradients from the rendering loss reach joint_offset/beta through the
    posing step -- see NeumanSMPLXDataset._world_bone_transforms's
    docstring. camera.bone_transforms is used only as a fallback for
    datasets that don't populate `body_model`/`align_matrix` (i.e. not
    NeumanSMPLXDataset)."""

    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        # List-wrapped so nn.Module.__setattr__ doesn't auto-register this as
        # a submodule -- body_model is OWNED (as a real submodule, so its
        # parameters land in the optimizer) by SMPLXTemplateOptimization
        # (src/models/pose_correction/cto.py) alone; VertexLBS only ever
        # calls read-only methods on it. Registering it here too would make
        # its parameters reachable from two different `.parameters()` calls,
        # which torch.optim.Adam rejects ("appear in more than one parameter
        # group") -- mirrors DirectPoseOptimization's `_mhr_model` pattern.
        self._body_model: list[object] = [metadata.get("body_model", None)]
        # Fallback only (datasets without a `body_model`/`align_matrix`, i.e.
        # not NeumanSMPLXDataset); moved to the right device lazily in
        # forward() rather than registered as a buffer, since VertexLBS has
        # no other parameters/buffers to co-locate it with.
        self.skinning_weights_static = torch.from_numpy(metadata["skinning_weights"]).float()

    @property
    def body_model(self):  # type: ignore[no-untyped-def]
        return self._body_model[0]

    def forward(self, gaussians: GaussianModel, iteration: int, camera: Camera) -> GaussianModel:
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]

        if self.body_model is not None and camera.align_matrix is not None:
            _, _, bone_transforms_local = self.body_model.pose_transforms(camera.rots)
            bone_transforms_local = bone_transforms_local[0]  # (J,4,4)
            tfs = torch.matmul(camera.align_matrix.unsqueeze(0), bone_transforms_local)  # (J,4,4)
            W = self.body_model.skinning_weights().to(xyz.device)
        else:
            tfs = camera.bone_transforms[0]
            W = self.skinning_weights_static.to(xyz.device)

        assert W.shape[0] == n_pts, (
            f"VertexLBS requires one Gaussian per canonical mesh vertex: got {n_pts} Gaussians "
            f"but {W.shape[0]} skinning-weight rows -- Scene.gaussians must be created via "
            "create_from_pcd(metadata['cano_verts'], ...) with no densification/pruning."
        )
        T_fwd = torch.einsum("vj,jab->vab", W, tfs)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        deformed_gaussians.rotation_precomp = rotation_bar

        return deformed_gaussians

    def regularization(self) -> dict[str, torch.Tensor]:
        return {}


def get_rigid_deform(cfg: DictConfig, metadata: MHRMetadata) -> RigidDeform:
    name = cfg.name
    model_dict = {
        "skinning_field": SkinningField,  # paper default
        "vertex_lbs": VertexLBS,  # GA-Avatar: 1 Gaussian per canonical mesh vertex
    }
    return model_dict[name](cfg, metadata)

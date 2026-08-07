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
        # Borrow the body model without registering it twice as a submodule;
        # it owns MHR's frozen rig and any trainable pose/CTO parameters.
        self._body_model: list[object] = [metadata.get("body_model", None)]

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

    @property
    def body_model(self):  # type: ignore[no-untyped-def]
        return self._body_model[0]

    def _pose_residuals(self, camera: Camera) -> torch.Tensor | None:
        model_params = getattr(camera, "mhr_model_params", None)
        if self.body_model is None or model_params is None:
            return None
        return self.body_model.pose_residuals(model_params)

    def _nearest_canonical_indices(self, xyz_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, idx, _ = knn_points(xyz_norm[None], self.cano_verts_norm[None], K=1)
        return idx.reshape(-1)

    def _nearest_canonical_indices_batched(self, xyz_norm_b: torch.Tensor) -> torch.Tensor:
        B = xyz_norm_b.shape[0]
        with torch.no_grad():
            cano_verts_b = self.cano_verts_norm.unsqueeze(0).expand(B, -1, -1).contiguous()
            _, idx, _ = knn_points(xyz_norm_b, cano_verts_b, K=1)
        return idx.reshape(B, -1)


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
        idx = self._nearest_canonical_indices(xyz_norm)
        default_w = self.skinning_weights_t[idx]
        offset_logits = self.lbs_network(xyz_norm)
        pts_W = self.softmax(torch.log(default_w + 1e-9) + offset_logits)
        return pts_W

    def predict_skinning_weights_batched(self, xyz_norm_b: torch.Tensor) -> torch.Tensor:
        """Batched form of predict_skinning_weights: xyz_norm_b is (B,N,3)
        (one canonical position set per view -- these can legitimately differ
        slightly across views since the non-rigid deformer runs first and its
        per-view delta is already baked into xyz_norm_b by the caller) and
        the KNN lookup + offset MLP both run once across the whole (B,N)
        batch instead of once per view. Not used when distill=True (no
        caller needs that combination yet)."""
        B, N, _ = xyz_norm_b.shape
        idx = self._nearest_canonical_indices_batched(xyz_norm_b)
        default_w = self.skinning_weights_t[idx]        # (B,N,J)
        offset_logits = self.lbs_network(xyz_norm_b)                # (B,N,J)
        pts_W = self.softmax(torch.log(default_w + 1e-9) + offset_logits)
        return pts_W

    def get_forward_transform_batched(self, xyz_b: torch.Tensor, tfs_b: torch.Tensor) -> torch.Tensor:
        assert not self.distill, "batched get_forward_transform not implemented for distill=True"
        pts_W = self.predict_skinning_weights_batched(xyz_b)  # (B,N,J)
        B, N, J = pts_W.shape
        T_fwd = torch.einsum("bnj,bjk->bnk", pts_W, tfs_b.reshape(B, J, 16)).view(B, N, 4, 4).float()
        return T_fwd

    def forward_batch(
        self, gaussians_list: list[GaussianModel], iteration: int, cameras: list[Camera]
    ) -> list[GaussianModel]:
        """Batched equivalent of forward(): the KNN skinning-weight lookup
        and its offset MLP run once across every view in the batch (one
        batched matmul each) instead of once per view, and the per-view bone
        transforms are applied via einsum instead of a Python loop of
        matmuls. `gaussians_list` is one already-non-rigid-deformed
        GaussianModel per view (see HashGridwithMLP.forward_batch) since
        their canonical positions can legitimately differ slightly."""
        B = len(gaussians_list)
        xyz_b = torch.stack([g.get_xyz for g in gaussians_list])          # (B,N,3)
        n_pts = xyz_b.shape[1]
        xyz_norm_b = self.aabb.normalize(xyz_b, sym=True)
        tfs_b = torch.stack([cam.bone_transforms for cam in cameras])      # (B,J,4,4)
        T_fwd_b = self.get_forward_transform_batched(xyz_norm_b, tfs_b)    # (B,N,4,4)

        homo_coord = torch.ones(B, n_pts, 1, dtype=torch.float32, device=xyz_b.device)
        x_hat_homo = torch.cat([xyz_b, homo_coord], dim=-1)                 # (B,N,4)
        x_bar = torch.einsum("bnij,bnj->bni", T_fwd_b, x_hat_homo)         # (B,N,4)

        if self.body_model is not None and all(
            getattr(camera, "mhr_model_params", None) is not None for camera in cameras
        ):
            model_params = torch.cat(
                [camera.mhr_model_params for camera in cameras], dim=0
            )
            residuals = self.body_model.pose_residuals(model_params)
            residual_idx = self._nearest_canonical_indices_batched(xyz_norm_b)
            batch_idx = torch.arange(B, device=residual_idx.device).unsqueeze(1)
            corrected_xyz = x_bar[:, :, :3] + residuals[batch_idx, residual_idx].to(
                device=x_bar.device, dtype=x_bar.dtype
            )
            x_bar = torch.cat([corrected_xyz, x_bar[:, :, 3:]], dim=-1)

        rotation_hat_b = torch.stack([build_rotation(g._rotation) for g in gaussians_list])  # (B,N,3,3)
        rotation_bar = torch.einsum("bnij,bnjk->bnik", T_fwd_b[:, :, :3, :3], rotation_hat_b)  # (B,N,3,3)

        out: list[GaussianModel] = []
        for b, g in enumerate(gaussians_list):
            deformed_gaussians = g.clone()
            deformed_gaussians.set_fwd_transform(T_fwd_b[b].detach())
            deformed_gaussians._xyz = x_bar[b, :, :3]
            deformed_gaussians.rotation_precomp = rotation_bar[b]
            out.append(deformed_gaussians)
        return out

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
        residuals = self._pose_residuals(camera)
        if residuals is not None:
            residual_idx = self._nearest_canonical_indices(xyz_norm)
            x_bar = x_bar + residuals[0, residual_idx]
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
        model_params = getattr(camera, "mhr_model_params", None)
        if self.body_model is not None and model_params is not None:
            residuals = self.body_model.pose_residuals(model_params)
            x_bar = x_bar + residuals[0].to(device=x_bar.device, dtype=x_bar.dtype)
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

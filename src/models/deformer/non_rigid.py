from __future__ import annotations

import pytorch3d.transforms as tf
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.constants import (
    GAUSSIAN_ROT_DIM,
    GAUSSIAN_SCALE_DIM,
    GAUSSIAN_XYZ_DIM,
    QUATERNION_IDENTITY_W,
    SCALE_EXP_MIN_EPS,
)
from src.dataset.mhr_native import MHRMetadata
from src.models.network_utils import (
    HannwCondMLP,
    HashGrid,
    HierarchicalPoseEncoder,
    VanillaCondMLP,
)
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel
from src.utils.general_utils import quaternion_multiply
from src.utils.loss_utils import laplacian_loss, sparse_laplacian_to_torch


class NonRigidDeform(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self, gaussians: GaussianModel, iteration: int, camera: Camera, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        raise NotImplementedError

class MLP(NonRigidDeform):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder, ktree_parents=metadata['joint_parents'])
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim: int = cfg.latent_dim
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = GAUSSIAN_XYZ_DIM
        d_out = GAUSSIAN_XYZ_DIM + GAUSSIAN_SCALE_DIM + GAUSSIAN_ROT_DIM
        self.feature_dim: int = cfg.feature_dim
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)
        self.aabb = metadata['aabb']

        self.delay: int = cfg.delay


    def forward(
        self, gaussians: GaussianModel, iteration: int, camera: Camera, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                deformed_gaussians.non_rigid_feature = torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda()
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx_val = len(self.frame_dict) - 1
            else:
                latent_idx_val = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx_val]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.scale_offset
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, SCALE_EXP_MIN_EPS))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.rot_offset
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = QUATERNION_IDENTITY_W # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            deformed_gaussians.non_rigid_feature = deltas[:, 10:]

        loss_reg: dict[str, torch.Tensor]
        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


class HannwMLP(NonRigidDeform):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder, ktree_parents=metadata['joint_parents'])
        # output dimension: position + scale + rotation
        self.mlp = HannwCondMLP(
            GAUSSIAN_XYZ_DIM, self.pose_encoder.n_output_dims,
            GAUSSIAN_XYZ_DIM + GAUSSIAN_SCALE_DIM + GAUSSIAN_ROT_DIM, cfg.mlp, dim_coord=GAUSSIAN_XYZ_DIM,
        )
        self.aabb = metadata['aabb']


    def forward(
        self, gaussians: GaussianModel, iteration: int, camera: Camera, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        deltas = self.mlp(xyz_norm, iteration, cond=pose_feat)

        if iteration < self.cfg.mlp.embedder.kick_in_iter:
            deltas = deltas * torch.zeros_like(deltas)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, -4:]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.scale_offset
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, SCALE_EXP_MIN_EPS))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.rot_offset
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = QUATERNION_IDENTITY_W  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            deformed_gaussians._rotation = quaternion_multiply(q1, q2)
        else:
            raise ValueError

        loss_reg: dict[str, torch.Tensor]
        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

class HashGridwithMLP(NonRigidDeform):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder, ktree_parents=metadata['joint_parents'])
        d_cond = self.pose_encoder.n_output_dims

        # add latent code
        self.latent_dim: int = cfg.latent_dim
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = GAUSSIAN_XYZ_DIM + GAUSSIAN_SCALE_DIM + GAUSSIAN_ROT_DIM
        self.feature_dim: int = cfg.feature_dim
        d_out += self.feature_dim

        self.aabb = metadata['aabb']
        self.hashgrid = HashGrid(cfg.hashgrid)
        self.mlp = VanillaCondMLP(self.hashgrid.n_output_dims, d_cond, d_out, cfg.mlp)

        self.delay: int = cfg.delay

    def forward(
        self, gaussians: GaussianModel, iteration: int, camera: Camera, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                deformed_gaussians.non_rigid_feature = (
                    torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda()
                )
            return deformed_gaussians, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx_val = len(self.frame_dict) - 1
            else:
                latent_idx_val = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx_val]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)

        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        deformed_gaussians = gaussians.clone()
        feature = self.hashgrid(xyz_norm)
        deltas = self.mlp(feature, cond=pose_feat)

        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]

        deformed_gaussians._xyz = gaussians._xyz + delta_xyz

        scale_offset = self.cfg.scale_offset
        if scale_offset == 'logit':
            deformed_gaussians._scaling = gaussians._scaling + delta_scale
        elif scale_offset == 'exp':
            deformed_gaussians._scaling = torch.log(torch.clamp_min(gaussians.get_scaling + delta_scale, SCALE_EXP_MIN_EPS))
        elif scale_offset == 'zero':
            delta_scale = torch.zeros_like(delta_scale)
            deformed_gaussians._scaling = gaussians._scaling
        else:
            raise ValueError

        rot_offset = self.cfg.rot_offset
        if rot_offset == 'add':
            deformed_gaussians._rotation = gaussians._rotation + delta_rot
        elif rot_offset == 'mult':
            q1 = delta_rot
            q1[:, 0] = QUATERNION_IDENTITY_W  # [1,0,0,0] represents identity rotation
            delta_rot = delta_rot[:, 1:]
            q2 = gaussians._rotation
            # deformed_gaussians._rotation = quaternion_multiply(q1, q2)
            deformed_gaussians._rotation = tf.quaternion_multiply(q1, q2)
        else:
            raise ValueError

        if self.feature_dim > 0:
            deformed_gaussians.non_rigid_feature = deltas[:, 10:]

        loss_reg: dict[str, torch.Tensor]
        if compute_loss:
            # regularization
            loss_xyz = torch.norm(delta_xyz, p=2, dim=1).mean()
            loss_scale = torch.norm(delta_scale, p=1, dim=1).mean()
            loss_rot = torch.norm(delta_rot, p=1, dim=1).mean()
            loss_reg = {
                'nr_xyz': loss_xyz,
                'nr_scale': loss_scale,
                'nr_rot': loss_rot
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg

class GAAvatarGeo(NonRigidDeform):
    """GA-Avatar's GeoNet: a base MLP(F_tri) -> static (delta_mu_f, s_f) and a
    pose-conditioned refine MLP(F_tri, theta) -> pose-dependent
    (delta_mu_p, delta_s_p). No rotation output -- GA-Avatar fixes every
    Gaussian's rotation at identity (see GaussianModel.training_setup's
    `fix_rotation_opacity`), unlike the MLP/HannwMLP/HashGridwithMLP variants
    above, which all predict a rotation delta because their Gaussians are NOT
    1:1 with mesh vertices.

    mu_bar (the CTO canonical template's vertex position) is read LIVE from
    `metadata['body_model'].rest_vertices()` every forward() call, not from
    `gaussians.get_xyz` -- CTO's {beta, joint_offset, face_offset} keep
    changing during training (see SMPLXTemplateOptimization), so mu_bar must
    track them, not the frozen point cloud `create_from_pcd` seeded Gaussians
    with at dataset-load time."""

    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg)
        # List-wrapped: body_model is owned (as a real nn.Module submodule)
        # by SMPLXTemplateOptimization alone -- see VertexLBS's identical
        # `_body_model` pattern for why (torch.optim.Adam rejects a param
        # reachable from two different `.parameters()` calls).
        self._body_model: list[object] = [metadata["body_model"]]
        self.aabb = metadata["aabb"]
        # Likewise list-wrapped: `triplane` is a real submodule of
        # GaussianConverter (see its __init__), which is where its
        # parameters get their own optimizer group; GAAvatarGeo and
        # GAAvatarRGB both just borrow the same instance to read from.
        self._triplane: list[object] = [metadata["triplane"]]

        d_feat = self.triplane.n_output_dims
        d_pose = int(cfg.pose_dim)
        # base MLP: F_tri -> (delta_mu_f(3), s_f(3)); unconditioned, so it
        # uses its own `mlp_base` config (cond_in: []) rather than sharing
        # `mlp_refine`'s (cond_in: [0]) -- VanillaCondMLP.forward asserts
        # cond is not None whenever cond_in is non-empty.
        self.base_mlp = VanillaCondMLP(d_feat, 0, 2 * GAUSSIAN_XYZ_DIM, cfg.mlp_base)
        # refine MLP: F_tri, theta -> (delta_mu_p(3), delta_s_p(3))
        self.refine_mlp = VanillaCondMLP(d_feat, d_pose, 2 * GAUSSIAN_XYZ_DIM, cfg.mlp_refine)
        self.delay: int = cfg.get("delay", 0)
        self.register_buffer("laplacian", sparse_laplacian_to_torch(metadata["laplacian"]))

    @property
    def body_model(self):  # type: ignore[no-untyped-def]
        return self._body_model[0]

    @property
    def triplane(self):  # type: ignore[no-untyped-def]
        return self._triplane[0]

    def forward(
        self, gaussians: GaussianModel, iteration: int, camera: Camera, compute_loss: bool = True
    ) -> tuple[GaussianModel, dict[str, torch.Tensor]]:
        mu_bar = self.body_model.rest_vertices()  # (V,3), live w.r.t. current CTO params
        xyz_norm = self.aabb.normalize(mu_bar, sym=True)
        f_tri = self.triplane(xyz_norm)

        base_out = self.base_mlp(f_tri)
        delta_mu_f, s_f = base_out[:, :3], base_out[:, 3:]

        if iteration < self.delay:
            delta_mu_p = torch.zeros_like(delta_mu_f)
            delta_s_p = torch.zeros_like(s_f)
        else:
            refine_out = self.refine_mlp(f_tri, cond=camera.rots)
            delta_mu_p, delta_s_p = refine_out[:, :3], refine_out[:, 3:]

        mu = mu_bar + delta_mu_f + delta_mu_p
        s = s_f + delta_s_p

        deformed_gaussians = gaussians.clone()
        deformed_gaussians._xyz = mu
        deformed_gaussians._scaling = s
        # GAAvatarRGB reads this (not `get_xyz`, which VertexLBS overwrites
        # with posed world-space coordinates right after this) so it samples
        # the shared triplane at the same canonical position GeoNet does.
        deformed_gaussians.canonical_xyz = mu
        # Rotation fixed identity: leave deformed_gaussians._rotation as
        # whatever GaussianModel.create_from_pcd initialized it to (identity
        # quaternion by construction) and never optimized (fix_rotation_opacity).

        loss_reg: dict[str, torch.Tensor]
        if compute_loss:
            loss_reg = {
                "ga_off_mu": torch.norm(delta_mu_f, p=2, dim=1).mean() + torch.norm(delta_mu_p, p=2, dim=1).mean(),
                "ga_off_scale": torch.norm(s_f, p=2, dim=1).mean() + torch.norm(delta_s_p, p=2, dim=1).mean(),
                "ga_lap_mu": laplacian_loss(delta_mu_f + delta_mu_p, self.laplacian),
                "ga_lap_scale": laplacian_loss(s_f + delta_s_p, self.laplacian),
            }
        else:
            loss_reg = {}
        return deformed_gaussians, loss_reg


def get_non_rigid_deform(cfg: DictConfig, metadata: MHRMetadata) -> NonRigidDeform:
    name = cfg.name
    model_dict = {
        "mlp": MLP,
        "hannw_mlp": HannwMLP,
        "hashgrid": HashGridwithMLP,  # paper default
        "ga_avatar_geo": GAAvatarGeo,  # GA-Avatar: triplane-conditioned dual-branch geometry
    }
    return model_dict[name](cfg, metadata)

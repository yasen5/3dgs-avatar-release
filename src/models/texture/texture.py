from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.constants import COV_UPPER_TRI_DIM, DIR_NORM_EPS, GAUSSIAN_XYZ_DIM, RGB_DIM, SH_DC_OFFSET
from src.dataset.mhr_native import MHRMetadata
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel
from src.models.network_utils import VanillaCondMLP
from src.utils.general_utils import build_rotation
from src.utils.sh_utils import augm_rots, eval_sh, eval_sh_bases


class ColorPrecompute(nn.Module):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

    def forward(self, gaussians: GaussianModel, camera: Camera) -> torch.Tensor:
        raise NotImplementedError

class SH2RGB(ColorPrecompute):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg, metadata)

    def forward(self, gaussians: GaussianModel, camera: Camera) -> torch.Tensor:
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            assert T_fwd is not None
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.view_noise
            if self.training and view_noise_scale > 0.:
                view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                          dtype=torch.float32,
                                          device=dir_pp.device).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, view_noise)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + DIR_NORM_EPS)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + SH_DC_OFFSET, 0.0)
        return colors_precomp

class ColorMLP(ColorPrecompute):
    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg, metadata)
        d_in = cfg.feature_dim

        self.use_xyz: bool = cfg.use_xyz
        self.use_cov: bool = cfg.use_cov
        self.use_normal: bool = cfg.use_normal
        self.sh_degree: int = cfg.sh_degree
        self.cano_view_dir: bool = cfg.cano_view_dir
        self.non_rigid_dim: int = cfg.non_rigid_dim
        self.latent_dim: int = cfg.latent_dim

        if self.use_xyz:
            d_in += GAUSSIAN_XYZ_DIM
        if self.use_cov:
            d_in += COV_UPPER_TRI_DIM # only upper triangle suffice
        if self.use_normal:
            d_in += GAUSSIAN_XYZ_DIM # quasi-normal by smallest eigenvector...
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim
        if self.latent_dim > 0:
            d_in += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_out = RGB_DIM
        self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)
        self.color_activation = nn.Sigmoid()

    def compose_input(self, gaussians: GaussianModel, camera: Camera) -> torch.Tensor:
        features = gaussians.get_features.squeeze(-1)
        n_points = features.shape[0]
        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            features = torch.cat([features, xyz_norm], dim=1)
        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)
        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
            features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                assert T_fwd is not None
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.view_noise
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + DIR_NORM_EPS)
            dir_embed = self.sh_embed(dir_pp_normalized)
            features = torch.cat([features, dir_embed], dim=1)
        if self.non_rigid_dim > 0:
            assert gaussians.non_rigid_feature is not None
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx_val = len(self.frame_dict) - 1
            else:
                latent_idx_val = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx_val]).long().to(features.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(features.shape[0], -1)
            features = torch.cat([features, latent_code], dim=1)

        return features


    def forward(self, gaussians: GaussianModel, camera: Camera) -> torch.Tensor:
        inp = self.compose_input(gaussians, camera)
        output = self.mlp(inp)
        color: torch.Tensor = self.color_activation(output)
        return color

    def compose_input_batch(self, gaussians_list: list[GaussianModel], cameras: list[Camera]) -> torch.Tensor:
        """Batched equivalent of compose_input: components that don't depend
        on the camera (the base per-Gaussian feature vector, unaffected by
        deformation) are computed once and broadcast; components that do
        (view direction, non-rigid feature, per-frame latent) are stacked
        per view. Returns (B,N,D_in)."""
        B = len(gaussians_list)
        base = gaussians_list[0]
        features = base.get_features.squeeze(-1)                   # (N,D) -- identical across views
        n_points = features.shape[0]
        parts = [features.unsqueeze(0).expand(B, n_points, -1)]

        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_b = torch.stack([aabb.normalize(g.get_xyz, sym=True) for g in gaussians_list])
            parts.append(xyz_b)
        if self.use_cov:
            cov_b = torch.stack([g.get_covariance() for g in gaussians_list])
            parts.append(cov_b)
        if self.use_normal:
            normals = []
            for g in gaussians_list:
                scale = g._scaling
                rot = build_rotation(g._rotation)
                normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
                normals.append(normal)
            parts.append(torch.stack(normals))
        if self.sh_degree > 0:
            dir_pps = []
            for g, camera in zip(gaussians_list, cameras):
                dir_pp = (g.get_xyz - camera.camera_center.repeat(n_points, 1))
                if self.cano_view_dir:
                    T_fwd = g.fwd_transform
                    assert T_fwd is not None
                    R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                    dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                    view_noise_scale = self.cfg.view_noise
                    if self.training and view_noise_scale > 0.:
                        view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                                  dtype=torch.float32,
                                                  device=dir_pp.device).transpose(0, 1)
                        dir_pp = torch.matmul(dir_pp, view_noise)
                dir_pps.append(dir_pp)
            dir_pp_b = torch.stack(dir_pps)                          # (B,N,3)
            dir_pp_normalized = dir_pp_b / (dir_pp_b.norm(dim=-1, keepdim=True) + DIR_NORM_EPS)
            dir_embed = self.sh_embed(dir_pp_normalized)             # (B,N,D_sh)
            parts.append(dir_embed)
        if self.non_rigid_dim > 0:
            for g in gaussians_list:
                assert g.non_rigid_feature is not None
            nrf = torch.stack([g.non_rigid_feature for g in gaussians_list])  # (B,N,non_rigid_dim)
            parts.append(nrf)
        if self.latent_dim > 0:
            idxs = [
                self.frame_dict[camera.frame_id] if camera.frame_id in self.frame_dict else len(self.frame_dict) - 1
                for camera in cameras
            ]
            latent_idx = torch.tensor(idxs, device=features.device).long()
            latent_code = self.latent(latent_idx)                    # (B, latent_dim)
            parts.append(latent_code.unsqueeze(1).expand(-1, n_points, -1))

        return torch.cat(parts, dim=-1)

    def forward_batch(self, gaussians_list: list[GaussianModel], cameras: list[Camera]) -> torch.Tensor:
        """Batched equivalent of forward(): one MLP call over the whole
        (B,N,D_in) input instead of one (N,D_in) call per view."""
        inp = self.compose_input_batch(gaussians_list, cameras)
        output = self.mlp(inp)
        color: torch.Tensor = self.color_activation(output)
        return color


class GAAvatarRGB(ColorPrecompute):
    """GA-Avatar's RGBNet: a base MLP(F_tri) -> static color logit c_f and a
    pose-conditioned refine MLP(F_tri, theta) -> pose-dependent color logit
    offset delta_c_p; c = (tanh(c_f + delta_c_p) + 1) / 2 -- see `forward`'s
    comment for why the activation is applied once, at the end, rather than
    to c_f alone, and why tanh (ExAvatar's convention) rather than sigmoid.
    Shares the same `metadata['triplane']` TriplaneEncoder instance as
    GAAvatarGeo (see GaussianConverter.__init__), matching the paper's single
    shared F_tri feeding both branches."""

    def __init__(self, cfg: DictConfig, metadata: MHRMetadata) -> None:
        super().__init__(cfg, metadata)
        # List-wrapped: `triplane` is a real submodule of GaussianConverter
        # (its own parameter/optimizer group lives there); GAAvatarRGB just
        # borrows the same instance to read from -- see GAAvatarGeo's
        # identical pattern for why this must not be a plain nn.Module attr.
        self._triplane: list[object] = [metadata["triplane"]]
        d_feat = self.triplane.n_output_dims
        d_pose = int(cfg.pose_dim)
        self.base_mlp = VanillaCondMLP(d_feat, 0, RGB_DIM, cfg.mlp_base)
        self.refine_mlp = VanillaCondMLP(d_feat, d_pose, RGB_DIM, cfg.mlp_refine)

    @property
    def triplane(self):  # type: ignore[no-untyped-def]
        return self._triplane[0]

    def forward(self, gaussians: GaussianModel, camera: Camera) -> torch.Tensor:
        # `gaussians.get_xyz` has already been posed to world space by
        # VertexLBS (rigid runs after non-rigid in Deformer.forward, see
        # deformer.py) by the time this runs -- normalizing THAT by the
        # canonical-mesh AABB would put every query far outside [-1,1],
        # which Triplane's grid_sample(padding_mode="border") clamps to a
        # near-constant per-frame value, killing gradient into RGBNet
        # (confirmed empirically: v4's RGBNet weights were bit-identical
        # from iteration 2000 through 12000 despite the last_layer_init fix).
        # `canonical_xyz` is GAAvatarGeo's live mu_bar + offsets, stashed
        # before VertexLBS overwrote `_xyz` -- the same quantity GeoNet
        # itself queries the triplane at, matching the paper's single
        # shared F_tri per canonical vertex.
        assert gaussians.canonical_xyz is not None, (
            "GAAvatarRGB requires a non-rigid deformer (e.g. ga_avatar_geo) "
            "that populates gaussians.canonical_xyz before rigid posing"
        )
        xyz_norm = self.metadata["aabb"].normalize(gaussians.canonical_xyz, sym=True)
        f_tri = self.triplane(xyz_norm)

        # Both branches stay in unconstrained logit space, combined additively,
        # with a single activation applied at the very end -- mirrors GeoNet's
        # scale output (s_f + delta_s_p, safe under its downstream exp()).
        # The previous clamp(sigmoid(c_f) + delta_c_p, 0, 1) applied sigmoid to
        # c_f BEFORE adding delta_c_p, then hard-clamped the sum: torch.clamp
        # has exactly-zero gradient for any excursion outside [0,1], and
        # confirmed empirically (v5, 2026-08-04) delta_c_p converges strongly
        # negative within the first few hundred iterations regardless of
        # last_layer_init, pushing c_f+delta_c_p negative for 100% of
        # Gaussian/channel pairs -- an irreversible, permanent gradient-zero
        # collapse. A smooth activation (never exactly zero gradient for
        # finite inputs) applied once at the end avoids that specific failure
        # mode.
        #
        # tanh (mapped to [0,1] via (tanh(x)+1)/2), not sigmoid: the paper
        # doesn't specify which; ExAvatar_RELEASE (common/nets/module.py's
        # HumanGaussian.forward) -- the closest available reference
        # implementation for this triplane-conditioned dual-branch avatar
        # architecture -- uses tanh, so we match that convention here rather
        # than guessing independently.
        c_f_logit = self.base_mlp(f_tri)
        delta_c_p = self.refine_mlp(f_tri, cond=camera.rots)

        color: torch.Tensor = (torch.tanh(c_f_logit + delta_c_p) + 1.0) / 2.0
        return color


def get_texture(cfg: DictConfig, metadata: MHRMetadata) -> ColorPrecompute:
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,  # paper default (via the "shallow_mlp" config preset)
        "ga_avatar_rgb": GAAvatarRGB,  # GA-Avatar: triplane-conditioned dual-branch appearance
    }
    return model_dict[name](cfg, metadata)

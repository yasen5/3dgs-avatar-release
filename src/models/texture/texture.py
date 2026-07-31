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


def get_texture(cfg: DictConfig, metadata: MHRMetadata) -> ColorPrecompute:
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,  # paper default (via the "shallow_mlp" config preset)
    }
    return model_dict[name](cfg, metadata)

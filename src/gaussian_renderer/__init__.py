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

from __future__ import annotations

import math
from typing import TypedDict

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from omegaconf import DictConfig

from src.scene import Scene
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel
from src.utils.graphics_utils import compute_vertex_normals


class RenderOutput(TypedDict):
    deformed_gaussian: GaussianModel
    render: torch.Tensor
    viewspace_points: torch.Tensor
    visibility_filter: torch.Tensor
    radii: torch.Tensor
    loss_reg: dict[str, torch.Tensor]
    opacity_render: torch.Tensor | None
    normal_render: torch.Tensor | None
    depth_render: torch.Tensor | None


def render(data: Camera,
           iteration: int,
           scene: Scene,
           pipe: DictConfig,
           bg_color : torch.Tensor,
           scaling_modifier: float = 1.0,
           override_color: torch.Tensor | None = None,
           compute_loss: bool = True,
           return_opacity: bool = False,
           return_normal: bool = False,
           return_depth: bool = False) -> RenderOutput:
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    opacity_image: torch.Tensor | None = None
    if return_opacity:
        opacity_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_image = opacity_image[:1]

    normal_image: torch.Tensor | None = None
    if return_normal:
        # GA-Avatar's L_n needs a rendered normal map: per-vertex mesh
        # normals (face-connectivity based, computed on `pc`'s ALREADY-posed
        # positions so they reflect this frame's articulation), rasterized
        # exactly like the opacity pass above -- one more colors_precomp swap
        # reusing the same deformed Gaussians, not a second deformation pass.
        # scene.metadata['faces'] is topology (pose/CTO-independent), so it's
        # safe to reuse verbatim regardless of which body_model backend built it.
        #
        # KNOWN GAP (currently harmless since lambda_normal=0 in
        # configs/option/ga_avatar.yaml until Sapiens normal maps are
        # generated for a sequence): these are WORLD-space normals.
        # /mnt/ssd2/tavatar_data_prep/run_sapiens_normals.py's output is
        # camera/view-space (standard for that kind of normal predictor).
        # Before enabling L_n, either rotate this by `data.R` into camera
        # space or rotate the loaded GT by `data.R.T` into world space in
        # scripts/train.py before calling normal_map_loss -- comparing them
        # as-is would silently penalize a geometrically-correct prediction.
        vertex_normals = compute_vertex_normals(means3D, scene.metadata['faces'])
        normal_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=vertex_normals,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    depth_image: torch.Tensor | None = None
    if return_depth:
        # Same reused-Gaussians pattern as the opacity/normal passes: per-
        # Gaussian camera-space depth (Z, from the world-to-view transform)
        # as colors_precomp, for GA-Avatar's L_d (depth_loss_local/_global).
        means3D_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=1)
        cam_space = means3D_homo @ data.world_view_transform
        depth_z = cam_space[:, 2:3].expand(-1, 3).contiguous()
        depth_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=depth_z,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        depth_image = depth_image[:1]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "normal_render": normal_image,
            "depth_render": depth_image,
            }

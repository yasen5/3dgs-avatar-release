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

from typing import TypedDict

import gsplat
import torch
from omegaconf import DictConfig

from src.scene import Scene
from src.scene.cameras import Camera
from src.scene.gaussian_model import GaussianModel
from src.utils.general_utils import build_scaling_rotation
from src.utils.graphics_utils import compute_vertex_normals


class RenderOutput(TypedDict):
    deformed_gaussian: GaussianModel
    render: torch.Tensor
    viewspace_points: torch.Tensor
    viewspace_batch_idx: int
    visibility_filter: torch.Tensor
    radii: torch.Tensor
    loss_reg: dict[str, torch.Tensor]
    opacity_render: torch.Tensor | None
    normal_render: torch.Tensor | None
    depth_render: torch.Tensor | None


def _full_covariance(pc: GaussianModel, scaling_modifier: float) -> torch.Tensor:
    """Unstripped (N,3,3) covariance -- gsplat's `covars` input wants the full
    matrix, unlike pc.get_covariance()'s upper-triangular-6 (cov3D_precomp)
    convention used by the old Inria rasterizer. Always incorporates
    rotation_precomp (the deformer's per-Gaussian LBS rotation matrix) when
    set, same as pc.get_covariance -- this repo runs with
    pipe.compute_cov3D_python=True everywhere, so that's the only path
    actually exercised.
    """
    rotation = pc.rotation_precomp if pc.rotation_precomp is not None else pc.get_rotation
    L = build_scaling_rotation(scaling_modifier * pc.get_scaling, rotation)
    return L @ L.transpose(1, 2)


def _camera_matrices(cam: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """(viewmat, K) in gsplat's world-to-camera / column-vector convention.

    Camera.world_view_transform is stored pre-transposed (see Camera.__init__,
    built from getWorld2View2(...).transpose(0, 1)) for the old Inria
    rasterizer's row-vector convention; transposing it back recovers the
    standard world-to-camera matrix gsplat expects.
    """
    if cam.K is None:
        raise ValueError(f"Camera {cam.frame_id!r} has no K; gsplat rendering requires intrinsics")
    viewmat = cam.world_view_transform.transpose(0, 1)
    K = torch.as_tensor(cam.K, dtype=torch.float32, device=viewmat.device)
    return viewmat, K


def render_batch(cams: list[Camera],
                  iteration: int,
                  scene: Scene,
                  pipe: DictConfig,
                  bg_color: torch.Tensor,
                  scaling_modifier: float = 1.0,
                  compute_loss: bool = True,
                  return_opacity: bool = False,
                  return_normal: bool = False,
                  return_depth: bool = False) -> list[RenderOutput]:
    """Render a batch of views with a single gsplat.rasterization() call.

    Each view is a different articulated pose of the same avatar (same
    Gaussian count N, different per-frame LBS deformation), so batching
    stacks each view's already-deformed Gaussian attributes along a leading
    batch dim (gsplat's "..." batch dims), rather than sharing one static
    scene across multiple cameras.

    This replaces the old per-view Python loop over diff_gaussian_rasterization
    calls. That backend hardcodes CUDA's legacy default stream for every
    kernel and CUB call and has a synchronous (blocking) cudaMemcpy in its
    forward pass, so an earlier attempt to overlap per-view rendering by
    giving each view its own torch.cuda.Stream corrupted its scratch buffers
    and produced NaN gradients in the backward pass (see git history on
    scripts/train.py for that experiment). gsplat threads
    at::cuda::getCurrentCUDAStream() through its kernels/CUB calls and has no
    such barrier -- but more to the point, it exposes a real batched kernel
    launch, so there's no concurrency hazard to route around at all; all
    views in `cams` are rasterized together in one launch.

    Background compositing note: bg_color only fills the RGB block; the
    extra normal/depth channels (return_normal/return_depth) composite
    against zero. Both are gated to lambda=0 in every current config, so
    this has no effect on any config actually in use.
    """
    # convert_gaussians_batch runs pose-correction + the rigid/non-rigid
    # deformer + the texture MLP as batched tensor ops across every view in
    # `cams` at once (one MHR query, one hashgrid encoding, one skinning-MLP
    # call, one texture-MLP call -- see GaussianConverter.forward_batch and
    # its callees) instead of a Python loop calling convert_gaussians() once
    # per view. That per-view loop used to be the actual bottleneck: only
    # the final rasterize call below was ever batched, while the deformer/
    # texture forward passes -- the more expensive part, since they run a
    # neural network over every Gaussian -- stayed serial regardless of
    # batch_size, capping GPU utilization well under the render call's real
    # ceiling (see hand_only profiling notes). pcs is still one GaussianModel
    # per view afterward, so everything below is unchanged.
    pcs, loss_reg_shared, colors = scene.convert_gaussians_batch(cams, iteration, compute_loss)
    loss_regs: list[dict[str, torch.Tensor]] = [loss_reg_shared for _ in cams]

    means = torch.stack([pc.get_xyz for pc in pcs])                             # (B,N,3)
    covars = torch.stack([_full_covariance(pc, scaling_modifier) for pc in pcs])  # (B,N,3,3)
    opacities = torch.stack([pc.get_opacity.squeeze(-1) for pc in pcs])         # (B,N)
    # colors already (B,N,3) from convert_gaussians_batch

    normals: list[torch.Tensor] | None = None
    if return_normal:
        # GA-Avatar's L_n target: per-vertex mesh normals on this frame's
        # already-posed Gaussian positions (see the original per-view
        # render()'s docstring note on this before it was ported here --
        # KNOWN GAP: these are world-space normals vs. camera-space GT).
        normals = [compute_vertex_normals(pc.get_xyz, scene.metadata['faces']) for pc in pcs]

    depths_z: list[torch.Tensor] | None = None
    if return_depth:
        depths_z = []
        for cam, pc in zip(cams, pcs):
            means3D_homo = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[:, :1])], dim=1)
            cam_space = means3D_homo @ cam.world_view_transform
            depths_z.append(cam_space[:, 2:3])  # (N,1) camera-space Z, GA-Avatar's L_d target

    extra_channels = []
    if normals is not None:
        extra_channels.append(torch.stack(normals))
    if depths_z is not None:
        extra_channels.append(torch.stack(depths_z))
    all_colors = torch.cat([colors, *extra_channels], dim=-1) if extra_channels else colors  # (B,N,D)
    D = all_colors.shape[-1]

    viewmats, Ks = [], []
    for cam in cams:
        vm, k = _camera_matrices(cam)
        viewmats.append(vm)
        Ks.append(k)
    viewmats_t = torch.stack(viewmats).unsqueeze(1)  # (B,1,4,4)
    Ks_t = torch.stack(Ks).unsqueeze(1)               # (B,1,3,3)

    B = len(cams)
    W = int(cams[0].image_width)
    H = int(cams[0].image_height)
    backgrounds = torch.zeros(B, 1, D, device=means.device, dtype=means.dtype)
    backgrounds[:, :, :3] = bg_color.view(1, 1, 3)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means=means,
        quats=None,  # type: ignore[arg-type]  -- unused whenever covars is given
        scales=None,  # type: ignore[arg-type]
        covars=covars,
        opacities=opacities,
        colors=all_colors,
        viewmats=viewmats_t,
        Ks=Ks_t,
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
        backgrounds=backgrounds,
    )
    # retain_grad must be called on this exact object: gsplat's own autograd
    # graph runs through it, so any reshape/slice of it taken *before*
    # backward would be a dead branch that autograd never visits. Densification
    # (add_densification_stats) reads means2d_parent.grad[batch_idx] after
    # loss.backward() instead of expecting a per-view tensor with its own grad.
    # Validation renders run under torch.no_grad(), so means2d has no
    # autograd history there and retain_grad() would raise. Training still
    # retains the exact gsplat tensor for densification statistics.
    if meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()
    means2d_parent = meta["means2d"]

    render_colors = render_colors.squeeze(1).permute(0, 3, 1, 2)  # (B,D,H,W)
    render_alphas = render_alphas.squeeze(1).squeeze(-1)          # (B,H,W)
    radii = meta["radii"].squeeze(1).amax(dim=-1)                 # (B,N)

    outputs: list[RenderOutput] = []
    for b in range(B):
        ch = 3
        normal_render = None
        if normals is not None:
            normal_render = render_colors[b, ch:ch + 3]
            ch += 3
        depth_render = None
        if depths_z is not None:
            depth_render = render_colors[b, ch:ch + 1]
            ch += 1

        outputs.append(RenderOutput(
            deformed_gaussian=pcs[b],
            render=render_colors[b, :3],
            viewspace_points=means2d_parent,
            viewspace_batch_idx=b,
            visibility_filter=radii[b] > 0,
            radii=radii[b],
            loss_reg=loss_regs[b],
            # gsplat's render_alphas is the true accumulated per-pixel alpha,
            # which is what the old opacity pass (a whole extra rasterizer
            # call with colors_precomp=ones) approximated -- exactly, in
            # fact, for every config here since all of them set
            # white_background: false (bg=0), so alpha*1 + (1-alpha)*0 == alpha.
            opacity_render=render_alphas[b:b + 1] if return_opacity else None,
            normal_render=normal_render,
            depth_render=depth_render,
        ))
    return outputs


def render(data: Camera,
           iteration: int,
           scene: Scene,
           pipe: DictConfig,
           bg_color: torch.Tensor,
           scaling_modifier: float = 1.0,
           override_color: torch.Tensor | None = None,
           compute_loss: bool = True,
           return_opacity: bool = False,
           return_normal: bool = False,
           return_depth: bool = False) -> RenderOutput:
    """Render a single view. Background tensor (bg_color) must be on GPU!

    Thin wrapper over render_batch([data], ...) so every existing single-view
    caller (scripts/render.py, render_from_ply.py, eval_masked_psnr.py,
    render_mask_only_before_after.py, train.py's validation()) keeps working
    unchanged; override_color is accepted but unused, matching the previous
    Inria-backed implementation (colors always come from
    scene.convert_gaussians, never a caller-supplied override).
    """
    return render_batch(
        [data], iteration, scene, pipe, bg_color,
        scaling_modifier=scaling_modifier,
        compute_loss=compute_loss,
        return_opacity=return_opacity,
        return_normal=return_normal,
        return_depth=return_depth,
    )[0]

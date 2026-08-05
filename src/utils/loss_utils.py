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

from math import exp
from typing import NamedTuple

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from torch.autograd import Variable

from src.scene.gaussian_model import GaussianModel


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = ((network_output - gt) ** 2).mean()
    return result


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class AiapLosses(NamedTuple):
    xyz: torch.Tensor
    covariance: torch.Tensor


def full_aiap_loss(gs_can: GaussianModel, gs_obs: GaussianModel, n_neighbors: int = 5) -> AiapLosses:
    xyz_can = gs_can.get_xyz
    xyz_obs = gs_obs.get_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_obs.get_covariance()

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return AiapLosses(xyz=loss_xyz, covariance=loss_cov)


def aiap_loss(
    x_canonical: torch.Tensor,
    x_deformed: torch.Tensor,
    n_neighbors: int = 5,
    nn_ix: torch.Tensor | None = None,
) -> torch.Tensor:
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:, 0, 1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:, 0, 1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss


# --- GA-Avatar losses --------------------------------------------------------

def sparse_laplacian_to_torch(laplacian: scipy.sparse.spmatrix, device: str | torch.device = "cpu") -> torch.Tensor:
    """One-time conversion of a scipy.sparse (V,V) mesh Laplacian (e.g. from
    `trimesh.smoothing.laplacian_calculation`, as stored in
    `metadata['laplacian']`) into a torch sparse COO tensor, for repeated
    `torch.sparse.mm` use in `laplacian_loss` every training step."""
    coo = laplacian.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def laplacian_loss(delta: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """GA-Avatar's L_lap: mesh-graph-Laplacian smoothness penalty on a
    per-vertex quantity (position or scale offset). `laplacian` is a torch
    sparse (V,V) tensor from `sparse_laplacian_to_torch`."""
    smoothed = torch.sparse.mm(laplacian, delta)
    result: torch.Tensor = smoothed.pow(2).sum(dim=1).mean()
    return result


def normal_map_loss(normal_pred: torch.Tensor, normal_gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """GA-Avatar's L_n: L1 + (1 - cosine) between rendered and Sapiens-
    precomputed normal maps, restricted to `valid_mask` (foreground)
    pixels. `normal_pred`/`normal_gt`: (3,H,W); `valid_mask`: (1,H,W) or
    (H,W) boolean/float."""
    mask = valid_mask.reshape(1, *valid_mask.shape[-2:]).to(dtype=torch.bool)
    mask3 = mask.expand_as(normal_pred)
    n_valid = mask.sum().clamp_min(1)

    l1_term = (normal_pred - normal_gt).abs()[mask3].sum() / (3 * n_valid)

    cos = F.cosine_similarity(normal_pred, normal_gt, dim=0, eps=1e-8).unsqueeze(0)
    cos_term = (1.0 - cos)[mask].sum() / n_valid

    result: torch.Tensor = l1_term + cos_term
    return result


def _patch_normalize(x: torch.Tensor, patch_size: int, eps: float = 1e-6) -> torch.Tensor:
    """(1,H,W) -> per-`patch_size`x`patch_size`-patch (mean, std)-normalized map,
    for the local depth term's ``D^p = (d - mean_p) / (std_p + eps)``."""
    c, h, w = x.shape
    pad_h = (-h) % patch_size
    pad_w = (-w) % patch_size
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))
    ph, pw = x_pad.shape[-2] // patch_size, x_pad.shape[-1] // patch_size
    patches = x_pad.reshape(c, ph, patch_size, pw, patch_size).permute(0, 1, 3, 2, 4).reshape(c, ph, pw, -1)
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True)
    normed = (patches - mean) / (std + eps)
    normed = normed.reshape(c, ph, pw, patch_size, patch_size).permute(0, 1, 3, 2, 4).reshape(c, h + pad_h, w + pad_w)
    return normed[:, :h, :w]


def depth_loss_local(
    depth_pred: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor, patch_size: int = 32
) -> torch.Tensor:
    """GA-Avatar's local depth term: per-patch mean/std-normalized L2,
    restricted to `valid_mask`. depth_pred/depth_gt: (1,H,W)."""
    mask = valid_mask.reshape(1, *valid_mask.shape[-2:]).to(dtype=torch.bool)
    d_pred_n = _patch_normalize(depth_pred, patch_size)
    d_gt_n = _patch_normalize(depth_gt, patch_size)
    result: torch.Tensor = ((d_pred_n - d_gt_n) ** 2)[mask].mean()
    return result


def depth_loss_global(
    depth_pred: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """GA-Avatar's global depth term: single whole-image std-normalized L2,
    restricted to `valid_mask`. depth_pred/depth_gt: (1,H,W)."""
    mask = valid_mask.reshape(1, *valid_mask.shape[-2:]).to(dtype=torch.bool)
    gt_std = depth_gt[mask].std() + eps
    d_pred_n = (depth_pred - depth_pred[mask].mean()) / gt_std
    d_gt_n = (depth_gt - depth_gt[mask].mean()) / gt_std
    result: torch.Tensor = ((d_pred_n - d_gt_n) ** 2)[mask].mean()
    return result

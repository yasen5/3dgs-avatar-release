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
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


class BasicPointCloud(NamedTuple):
    points: npt.NDArray[np.floating[Any]]
    colors: npt.NDArray[np.floating[Any]]
    normals: npt.NDArray[np.floating[Any]]


def geom_transform_points(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R: npt.NDArray[np.floating[Any]], t: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.astype(np.float32)


def getWorld2View2(
    R: npt.NDArray[np.floating[Any]],
    t: npt.NDArray[np.floating[Any]],
    translate: npt.NDArray[np.floating[Any]] = np.array([.0, .0, .0]),
    scale: float = 1.0,
) -> npt.NDArray[np.float32]:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt.astype(np.float32)


def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def compute_vertex_normals(verts: torch.Tensor, faces: npt.NDArray[np.integer[Any]]) -> torch.Tensor:
    """Per-vertex normals from face connectivity (area-weighted face-normal
    average at each vertex), for GA-Avatar's rendered normal map (L_n) --
    only meaningful when `verts` is in mesh-vertex order (i.e. GA-Avatar's
    VertexLBS mode, one Gaussian per canonical mesh vertex), not for
    free-form/densified Gaussian point clouds."""
    faces_t = torch.as_tensor(faces, dtype=torch.long, device=verts.device)
    v0, v1, v2 = verts[faces_t[:, 0]], verts[faces_t[:, 1]], verts[faces_t[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # unnormalized -- magnitude = 2x triangle area

    vertex_normals = torch.zeros_like(verts)
    for i in range(3):
        vertex_normals.index_add_(0, faces_t[:, i], face_normals)

    return F.normalize(vertex_normals, dim=1, eps=1e-8)

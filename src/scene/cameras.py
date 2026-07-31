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

import copy as copy_module
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera:
    def __init__(
        self,
        frame_id: str,
        cam_id: int,
        R: npt.NDArray[np.floating[Any]],
        T: npt.NDArray[np.floating[Any]],
        FoVx: float,
        FoVy: float,
        image: torch.Tensor,
        mask: torch.Tensor,
        gt_alpha_mask: torch.Tensor | None,
        image_name: str,
        data_device: str,
        rots: torch.Tensor,
        Jtrs: torch.Tensor,
        bone_transforms: torch.Tensor,
        K: npt.NDArray[np.floating[Any]] | None = None,
        colmap_id: int | None = None,
        uid: int | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.cam_id = cam_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image = image
        self.mask = mask
        self.gt_alpha_mask = gt_alpha_mask
        self.image_name = image_name
        self.data_device = data_device
        self.rots = rots
        self.Jtrs = Jtrs
        self.bone_transforms = bone_transforms
        self.K = K
        self.colmap_id = colmap_id
        self.uid = uid

        self.trans: npt.NDArray[np.floating[Any]] = np.array([0.0, 0.0, 0.0])
        self.scale: float = 1.0

        self.original_image: torch.Tensor = self.image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width: int = self.original_image.shape[2]
        self.image_height: int = self.original_image.shape[1]
        self.original_mask: torch.Tensor = self.mask.float().to(self.data_device)

        self.zfar: float = 100.0
        self.znear: float = 0.01

        self.world_view_transform: torch.Tensor = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)
        ).transpose(0, 1).cuda()
        self.projection_matrix: torch.Tensor = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        self.full_proj_transform: torch.Tensor = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center: torch.Tensor = self.world_view_transform.inverse()[3, :3]

        self.rots = self.rots.to(self.data_device)
        self.Jtrs = self.Jtrs.to(self.data_device)
        self.bone_transforms = self.bone_transforms.to(self.data_device)

    def copy(self) -> Camera:
        return copy_module.copy(self)

    def update(
        self,
        *,
        rots: torch.Tensor | None = None,
        bone_transforms: torch.Tensor | None = None,
    ) -> None:
        if rots is not None:
            self.rots = rots
        if bone_transforms is not None:
            self.bone_transforms = bone_transforms

    def merge(self, cam: Camera) -> None:
        self.frame_id = cam.frame_id
        self.rots = cam.rots.detach()
        self.Jtrs = cam.Jtrs.detach()
        self.bone_transforms = cam.bone_transforms.detach()

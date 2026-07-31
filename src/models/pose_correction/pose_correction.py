from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.body_models import mhr_lbs
from src.body_models.mhr_utils import local_joint_rotmats
from src.dataset.mhr_native import MHRMetadata
from src.scene.cameras import Camera


class NoPoseCorrection(nn.Module):
    def __init__(self, config: DictConfig, metadata: MHRMetadata | None = None) -> None:
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera: Camera, iteration: int) -> tuple[Camera, dict[str, torch.Tensor]]:
        return camera, {}

    def regularization(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}


class DirectPoseOptimization(nn.Module):
    """`model_params` (204, the whole skeletal/articulated state --
    global_trans+global_rot+body_pose) is a learnable nn.Embedding per frame,
    photometrically refined by the real Gaussian rendering loss during
    training, initialized from prepare_mhr_keyframe_track.py's
    SAM3D-Body-direct-at-keyframes/SLERP-interpolated-elsewhere track.
    `shape_params` (45, MHR's non-skeletal body-identity params) is a frozen
    buffer everywhere -- body shape is one physical identity that does not
    change frame-to-frame, so it is never touched by any gradient here.

    Jtrs (the pose-encoder's canonical-joint-position context feature) is
    deliberately NOT recomputed here -- the fixed big-pose Jtrs normalization
    computed once in dataset/mhr_native.py stays self-consistent enough;
    recomputing it every iteration would need an extra full MHR big-pose
    forward pass for a negligible benefit.
    """

    joint_parents: torch.Tensor
    big_pose_joint_pos: torch.Tensor
    big_pose_joint_rotmat: torch.Tensor
    shape_params_all: torch.Tensor
    cam_t_all: torch.Tensor
    model_params_init: torch.Tensor

    def __init__(self, config: DictConfig, metadata: MHRMetadata | None = None) -> None:
        super(DirectPoseOptimization, self).__init__()
        assert metadata is not None
        self.config = config
        self.frame_dict = metadata['frame_dict']

        # list-wrapped so nn.Module.__setattr__ doesn't auto-register this
        # shared, already-frozen (requires_grad_(False) at load time) MHR
        # TorchScript module as a submodule -- it would otherwise show up in
        # self.parameters() (used both for the optimizer's param group and
        # for grad-clipping) as several hundred inert always-zero-grad tensors.
        self._mhr_model: list[torch.jit.ScriptModule] = [metadata['mhr_model']]

        self.register_buffer('joint_parents', torch.from_numpy(np.asarray(metadata['joint_parents'])).long())
        self.register_buffer('big_pose_joint_pos', metadata['big_pose_joint_pos'].clone())
        self.register_buffer('big_pose_joint_rotmat', metadata['big_pose_joint_rotmat'].clone())
        self.register_buffer('shape_params_all', metadata['shape_params_all'].clone())
        self.register_buffer('cam_t_all', metadata['cam_t_all'].clone())
        self.register_buffer('model_params_init', metadata['model_params_all'].clone())
        self.model_params: nn.Embedding = nn.Embedding.from_pretrained(metadata['model_params_all'].clone(), freeze=False)

    def forward(self, camera: Camera, iteration: int) -> tuple[Camera, dict[str, torch.Tensor]]:
        frame = camera.frame_id
        if frame not in self.frame_dict:
            return camera, {}
        if iteration < self.config.get('delay', 0):
            return camera, {}

        idx = torch.tensor([self.frame_dict[frame]], device=self.model_params.weight.device).long()
        shape = self.shape_params_all[idx]  # (1,45), frozen
        model_params = self.model_params(idx)  # (1,204), learnable
        cam_t = self.cam_t_all[idx][0]  # (3,)

        out = mhr_lbs.mhr_query(self._mhr_model[0], shape, model_params, device=str(shape.device))
        joint_pos = out['joint_pos']  # (1,127,3)
        joint_rotmat = out['joint_rotmat']  # (1,127,3,3)

        pose_rot = local_joint_rotmats(joint_rotmat, self.joint_parents)[0].reshape(-1, 9)  # (127,9)

        bone_transforms = mhr_lbs.joint_relative_transforms(
            joint_pos, joint_rotmat,
            self.big_pose_joint_pos.unsqueeze(0), self.big_pose_joint_rotmat.unsqueeze(0),
        )[0].clone()  # (127,4,4)
        bone_transforms[:, :3, 3] = bone_transforms[:, :3, 3] + cam_t

        updated_camera = camera.copy()
        updated_camera.update(
            rots=pose_rot.unsqueeze(0),
            bone_transforms=bone_transforms,
        )

        # keeps the optimized pose close to the tracked initialization, so
        # single noisy frames (motion blur, an occluder) don't chase
        # implausible poses to minimize local photometric loss.
        loss_pose = (model_params - self.model_params_init[idx]).square().mean()
        return updated_camera, {'pose': loss_pose}

    def regularization(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {}


def get_pose_correction(cfg: DictConfig, metadata: MHRMetadata) -> nn.Module:
    name = cfg.name
    model_dict = {
        "none": NoPoseCorrection,
        "direct": DirectPoseOptimization,
    }
    return model_dict[name](cfg, metadata)

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

from typing import Any, NamedTuple, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL.Image import Image as PILImage

from src.scene.cameras import Camera
from src.utils.general_utils import PILtoTorch
from src.utils.graphics_utils import fov2focal

WARNED = False


class CameraInfo(NamedTuple):
    image: PILImage
    mask: PILImage
    uid: int
    frame_id: str
    cam_id: int
    R: npt.NDArray[np.floating[Any]]
    T: npt.NDArray[np.floating[Any]]
    FovX: float
    FovY: float
    image_name: str
    rots: torch.Tensor
    Jtrs: torch.Tensor
    bone_transforms: torch.Tensor


class CameraJSONEntry(TypedDict):
    id: int
    img_name: str
    width: int
    height: int
    position: list[float]
    rotation: list[list[float]]
    fy: float
    fx: float


def loadCam(args: DictConfig, id: int, cam_info: CameraInfo, resolution_scale: float) -> Camera:
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_mask = PILtoTorch(cam_info.mask, resolution)
    gt_mask = resized_mask[:1, ...] != 0

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, frame_id=cam_info.frame_id, cam_id=cam_info.cam_id,
                  R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, mask=gt_mask, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  rots=cam_info.rots, Jtrs=cam_info.Jtrs, bone_transforms=cam_info.bone_transforms)

def cameraList_from_camInfos(
    cam_infos: list[CameraInfo], resolution_scale: float, args: DictConfig
) -> list[Camera]:
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id: int, camera: Camera) -> CameraJSONEntry:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry: CameraJSONEntry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.image_width,
        'height' : camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.image_height),
        'fx' : fov2focal(camera.FoVx, camera.image_width),
    }
    return camera_entry

# compute image space normal
def get_homo_2d(height: int, width: int) -> npt.NDArray[np.floating[Any]]:
    Y, X = np.meshgrid(np.arange(height, dtype=np.float32),
                       np.arange(width, dtype=np.float32),
                       indexing='ij')
    xy = np.stack([X, Y], axis=-1)
    homo_ones = np.ones((height, width, 1), dtype=np.float32)
    homo_2d: npt.NDArray[np.floating[Any]] = np.concatenate((xy, homo_ones), axis=2)
    return homo_2d

def get_inverse_intrinsic(K: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    K_inv = K.copy()
    K_inv[0, 0] = 1. / K[0, 0]
    K_inv[1, 1] = 1. / K[1, 1]
    K_inv[0, 2] = -K[0, 2] / K[0, 0]
    K_inv[1, 2] = -K[1, 2] / K[0, 0]
    return K_inv

def compute_normal_image(depth: torch.Tensor, fg_mask: torch.Tensor, camera: Camera) -> torch.Tensor:
    depth = depth.permute(1, 2, 0)

    height = camera.image_height
    width = camera.image_width

    homo_2d = get_homo_2d(height, width)
    K = camera.K
    assert K is not None
    K_inv = get_inverse_intrinsic(K)
    uv_np = np.dot(homo_2d.reshape([-1, 3]), K_inv.T).reshape([height, width, 3])
    uv = torch.from_numpy(uv_np).to(depth.device)
    cam_ray_dir = F.normalize(uv, p=2, dim=-1)

    pred_points = cam_ray_dir * depth

    zs = pred_points[:, :, 2]
    xs = pred_points[:, :, 0]
    ys = pred_points[:, :, 1]
    eps = 1e-10

    zy = (zs[1:, :] - zs[:-1, :]) / (ys[1:, :] - ys[:-1, :] + eps)
    zx = (zs[:, 1:] - zs[:, :-1]) / (xs[:, 1:] - xs[:, :-1] + eps)

    ny = torch.cat([-zy, torch.zeros(1, width, device=zy.device)], dim=0)
    nx = torch.cat([-zx, torch.zeros(height, 1, device=zx.device)], dim=1)
    nz = torch.ones(height, width, device=depth.device, dtype=torch.float32)
    pred_normals = torch.stack([nx, ny, nz], dim=0)

    n = torch.linalg.norm(pred_normals, dim=0, keepdim=True)
    pred_normals = pred_normals / n

    pred_normals[:, ~fg_mask] = -1
    result: torch.Tensor = ((pred_normals + 1) / 2.0).clip(0.0, 1.0)

    return result


def _update_extrinsics(
        extrinsics: npt.NDArray[np.floating[Any]],
        angle: float,
        trans: npt.NDArray[np.floating[Any]] | None = None,
        rotate_axis: str = 'y') -> npt.NDArray[np.floating[Any]]:
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle

    rotate_coord = {
        'x': 0, 'y': 1, 'z': 2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos)
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans

    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E


class FreeviewCamParams(TypedDict):
    K: npt.NDArray[np.floating[Any]]
    D: npt.NDArray[np.floating[Any]]
    R: npt.NDArray[np.floating[Any]]
    T: npt.NDArray[np.floating[Any]]


class FreeviewCameraSet(NamedTuple):
    all_cam_names: list[str]
    cams: dict[str, FreeviewCamParams]


def freeview_camera(camera: FreeviewCamParams, trans: npt.NDArray[np.floating[Any]] | None,
                    total_frames: int = 100,
                    rotate_axis: str = 'z',
                    inv_angle: bool = False) -> FreeviewCameraSet:

    cam_names = [str(cam_name) for cam_name in range(total_frames + 1)]
    all_cam_params: dict[str, FreeviewCamParams] = {}
    for frame_idx, cam_name in enumerate(cam_names):
        Ri = np.array(camera['R'], np.float32)
        Ti = np.array(camera['T'], np.float32)
        Ei = np.eye(4)
        Ei[:3,:3] = Ri
        Ei[:3,3:] = Ti

        angle = 2 * np.pi * (frame_idx / total_frames)
        if inv_angle:
            angle = -angle
        Eo = _update_extrinsics(Ei, angle, trans, rotate_axis)

        Ro = Eo[:3,:3]
        To = Eo[:3,3:]
        cam_params: FreeviewCamParams = {
            'K': camera['K'],
            'D': camera['D'],
            'R': Ro,
            'T': To,
        }

        all_cam_params[cam_name] = cam_params
    return FreeviewCameraSet(all_cam_names=cam_names, cams=all_cam_params)

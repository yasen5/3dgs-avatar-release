"""MHR-native dataset reader.

Reads either the {train,test}/mhr_parms.pth + images/ + valid_masks/ layout
produced by pipelines/texture_appearance/threedgs_avatar_mhr/prepare_mhr_keyframe_track.py,
or the ZJU-MoCap layout produced by the current MHR preparation run:
{root}/Camera_B1/{images,masks,results/raw}.  The ZJU reader is
explicitly single-camera; it never discovers or mixes the other Camera_B*
directories.

The posed mesh + per-joint transforms come directly from MHR's own forward pass
(body_models/mhr_lbs.py, symlinked to the shared, byte-verified adapter in the
sibling gauhuman_baseline project) -- there is no SMPL-style joint regressor or
blendshape reimplementation needed; see that module's docstring.

Masking: "mask out non-skin except the head" is implemented as background
compositing (not a separate loss-gate field) -- valid_masks/ (skin | full head
silhouette incl. hair) is used as BOTH the alpha-loss target (this dataset's
`original_mask`) and the region composited over the true image; every other
on-body pixel (clothing) is composited to the background color exactly like
true background, so Gaussians are pushed to zero opacity there rather than
asked to reproduce clothing texture.
"""
from __future__ import annotations

import os
from typing import Any, TypedDict
from typing_extensions import NotRequired, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from src.body_models import mhr_lbs
from src.body_models.mhr_body_model import MHRBodyModel
from src.body_models.metadata import CanonicalMetadata, ModelMetadata
from src.body_models.mhr_utils import build_big_pose_model_params, local_joint_rotmats
from src.constants import CAMERAS_EXTENT, MHR_MODEL_PARAMS_DIM, MHR_SHAPE_DIM
from src.dataset.training_interface import is_hand_dataset
from src.scene.cameras import Camera
from src.utils.dataset_utils import AABB, fetchPly, storePly
from src.utils.graphics_utils import BasicPointCloud, focal2fov

# Backward-compat aliases: these two TypedDicts used to be defined here and
# were MHR-only. They now live in src.body_models.metadata as the generic
# CanonicalMetadata/ModelMetadata (used by both the MHR and SMPL-X backends);
# re-exported under their original names so every existing MHR call site
# (`MHRCanonicalMetadata`, `MHRMetadata`) keeps working unmodified.
MHRCanonicalMetadata: TypeAlias = CanonicalMetadata
MHRMetadata: TypeAlias = ModelMetadata


class MHRParms(TypedDict):
    shape_params: torch.Tensor
    model_params: torch.Tensor
    cam_t: torch.Tensor
    focal_length: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    frame_ids: list[str]
    # only present in the {train,test}/mhr_parms.pth layout, and even there
    # only for keyframe-tracked runs -- see _load_prepared_source's `in` check.
    is_keyframe: NotRequired[torch.Tensor]


class RawFrame(TypedDict):
    """CPU-only frame payload produced by MHRNativeDataset.load_raw -- the part
    of a Camera that's safe to build inside a DataLoader worker subprocess
    (see load_raw's docstring for why the rest of Camera construction isn't)."""
    frame_id: str
    K: npt.NDArray[np.floating[Any]]
    FoVx: float
    FoVy: float
    image: torch.Tensor
    mask: torch.Tensor
    rots: torch.Tensor
    bone_transforms: torch.Tensor


class MHRNativeDataset(Dataset[Camera]):
    def __init__(self, cfg: DictConfig, split: str = 'train') -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root_dir: str = cfg.root_dir
        configured_hand_only = cfg.get("hand_only", None)
        self.hand_only = is_hand_dataset(
            self.root_dir,
            None if configured_hand_only is None else bool(configured_hand_only),
        )
        hand_side = cfg.get("hand_side", None)
        self.hand_side: str | None = None if hand_side is None else str(hand_side)
        if self.hand_side is not None:
            if not self.hand_only:
                raise ValueError("dataset.hand_side requires dataset.hand_only=True")
            if self.hand_side not in ("left", "right"):
                raise ValueError(
                    f"dataset.hand_side must be 'left' or 'right', got {self.hand_side!r}"
                )
        self.hand_crop_padding = float(cfg.get("hand_crop_padding", 0.25))
        self.white_bg: bool = cfg.white_background
        self.device: str = cfg.mhr_device
        self.source_layout: str = cfg.source_layout

        self.frame_ids: list[str] = []
        self._image_paths: dict[str, str] = {}
        self._mask_paths: dict[str, str] = {}
        self._depth_dir: str = ""
        self.camera_names: list[str] = []
        self.is_keyframe: npt.NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self.parms: MHRParms
        self.data_dir: str
        self.camera_name: str
        self.camera_dir: str

        if self.source_layout == 'zju_mocap':
            self._load_zju_mocap_source()
        elif self.source_layout == 'prepared':
            self._load_prepared_source()
        else:
            raise ValueError(
                "dataset.source_layout must be 'prepared' or 'zju_mocap', "
                f"got {self.source_layout!r}"
            )

        self.mhr_model = mhr_lbs.load_mhr(cfg.mhr_model_path, device=self.device)
        state_dict = mhr_lbs.mhr_state_dict(cfg.mhr_model_path)
        # Keep the native MHR model/query fields below for existing MHR-only
        # consumers, but expose the exact same initialized mesh abstraction as
        # the SMPL-X dataset. The adapter reuses this already-loaded rig rather
        # than loading a second TorchScript copy.
        first_shape = self.parms["shape_params"][0].float().reshape(-1).tolist()
        first_model_params = self.parms["model_params"][0].float().reshape(-1)
        canonical_model_params = build_big_pose_model_params(first_model_params).tolist()
        body_model_cfg = OmegaConf.create({
            "model_path": cfg.mhr_model_path,
            "device": self.device,
            "shape_params": first_shape,
            "canonical_model_params": canonical_model_params,
            "learn_shape": False,
        })
        self.body_model = MHRBodyModel(
            body_model_cfg,
            model=self.mhr_model,
            state_dict=state_dict,
        )
        self.body_model.set_hand_only(self.hand_only, side=self.hand_side)
        self.faces = self.body_model.faces()
        self.dense_skinning_weights = self.body_model.skinning_weights().numpy()
        self.joint_parents = torch.from_numpy(self.body_model.joint_parents())

        self.metadata: MHRCanonicalMetadata | MHRMetadata
        self.get_metadata()

        # Per-frame pose (joint rotations + bone transforms) only depends on
        # shape/model params + cam_t, never on the image -- so it is computed
        # once here as a single batched CUDA forward pass instead of once per
        # __getitem__ call. This keeps __getitem__ CPU-only (cv2 image/mask
        # decode + tensor bookkeeping), which is required for it to be safely
        # farmed out to torch DataLoader worker subprocesses (num_workers>0);
        # touching a CUDA model from a forked worker is unsupported.
        self.pose_rot_all: torch.Tensor
        self.bone_transforms_all: torch.Tensor
        self._precompute_pose_data()

        self.preload: bool = cfg.preload
        self.cameras: list[Camera] = []
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

        self._depth_dir = os.path.join(self.data_dir, "sapiens_depth")

    def _precompute_pose_data(self) -> None:
        """Batched replacement for the per-item mhr_query + joint_relative_transforms
        calls: runs the MHR forward pass over all frames (chunked to bound GPU
        memory) once, and stores the resulting per-frame pose_rot/bone_transforms
        as CPU tensors for __getitem__ to index into."""
        n = len(self.frame_ids)
        shape_params_all = self.parms["shape_params"].float()
        model_params_all = self.parms["model_params"].float()
        cam_t_all = self.parms["cam_t"].float()

        big_pose_joint_pos = self.metadata["big_pose_joint_pos"]  # (127,3)
        big_pose_joint_rotmat = self.metadata["big_pose_joint_rotmat"]  # (127,3,3)

        chunk_size = 256
        pose_rot_chunks: list[torch.Tensor] = []
        bone_transforms_chunks: list[torch.Tensor] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            b = end - start
            with torch.no_grad():
                out = mhr_lbs.mhr_query(
                    self.mhr_model,
                    shape_params_all[start:end].to(self.device),
                    model_params_all[start:end].to(self.device),
                    device=self.device,
                )
            joint_pos = out["joint_pos"].cpu()  # (b,127,3)
            joint_rotmat = out["joint_rotmat"].cpu()  # (b,127,3,3)

            pose_rot = local_joint_rotmats(joint_rotmat, self.joint_parents)
            pose_rot = pose_rot.reshape(b, -1, 9)  # (b,127,9)

            rest_pos = big_pose_joint_pos.unsqueeze(0).expand(b, -1, -1)
            rest_rotmat = big_pose_joint_rotmat.unsqueeze(0).expand(b, -1, -1, -1)
            bone_transforms = mhr_lbs.joint_relative_transforms(
                joint_pos, joint_rotmat, rest_pos, rest_rotmat,
            )  # (b,127,4,4)
            bone_transforms[:, :, :3, 3] += cam_t_all[start:end].unsqueeze(1)

            pose_rot_chunks.append(pose_rot)
            bone_transforms_chunks.append(bone_transforms)

        self.pose_rot_all = torch.cat(pose_rot_chunks, dim=0).float()
        self.bone_transforms_all = torch.cat(bone_transforms_chunks, dim=0).float()

    def _load_prepared_source(self) -> None:
        """Load the native train/test directory layout."""
        data_split = 'train' if self.split == 'train' else 'test'
        self.data_dir = os.path.join(self.root_dir, data_split)
        self.parms = torch.load(
            os.path.join(self.data_dir, "mhr_parms.pth"), weights_only=False
        )
        self.frame_ids = list(self.parms["frame_ids"])
        self._image_paths = {
            frame_id: os.path.join(self.data_dir, "images", f"{frame_id}.png")
            for frame_id in self.frame_ids
        }
        self._mask_paths = {
            frame_id: os.path.join(self.data_dir, "valid_masks", f"{frame_id}.png")
            for frame_id in self.frame_ids
        }
        self.camera_names = ['native']
        self.is_keyframe = self.parms["is_keyframe"].numpy() if "is_keyframe" in self.parms else np.zeros(
            len(self.frame_ids), dtype=bool)

    def _load_zju_mocap_source(self) -> None:
        """Load one camera from a prepared ZJU-MoCap subject directory.

        The per-frame raw files are the source of truth for MHR pose, shape,
        camera translation, focal length, and image dimensions.  This keeps
        the loader in the same camera-space convention as ``mhr_lbs.mhr_query``
        and avoids relying on the multi-view annots.npy extrinsics.
        """
        camera_name = str(self.cfg.camera_name)
        if not camera_name.startswith('Camera_B'):
            raise ValueError(
                "dataset.camera_name must name one ZJU camera such as "
                f"'Camera_B1', got {camera_name!r}"
            )

        self.camera_name = camera_name
        self.camera_names = [camera_name]
        self.camera_dir = os.path.join(self.root_dir, camera_name)
        raw_dir = os.path.join(self.camera_dir, 'results', 'raw')
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"MHR raw-fit directory not found: {raw_dir}")

        # Confirm the requested camera is part of this subject's annotation
        # table, but do not use annots.npy to load the other cameras.
        annots_path = os.path.join(self.root_dir, 'annots.npy')
        if not os.path.isfile(annots_path):
            raise FileNotFoundError(f"ZJU annotation file not found: {annots_path}")
        annots = np.load(annots_path, allow_pickle=True).item()
        annotated_cameras = {
            os.path.basename(os.path.dirname(path))
            for path in annots['ims'][0]['ims']
        }
        if camera_name not in annotated_cameras:
            raise ValueError(
                f"{camera_name} is not present in {annots_path}; "
                f"available cameras: {sorted(annotated_cameras)}"
            )

        raw_paths = sorted(
            [p for p in os.listdir(raw_dir) if p.endswith('.npz')],
            key=lambda name: int(os.path.splitext(name)[0]),
        )
        if not raw_paths:
            raise FileNotFoundError(f"No MHR raw fits found in {raw_dir}")

        image_dir = os.path.join(self.camera_dir, 'images')
        mask_dir = os.path.join(
            self.camera_dir,
            "hand_masks" if self.hand_only else str(self.cfg.mask_dir),
        )
        frame_ids: list[str] = []
        shape_params: list[npt.NDArray[np.floating[Any]]] = []
        model_params: list[npt.NDArray[np.floating[Any]]] = []
        cam_t: list[npt.NDArray[np.floating[Any]]] = []
        focal_lengths: list[float] = []
        widths: list[int] = []
        heights: list[int] = []
        self._image_paths = {}
        self._mask_paths = {}

        for raw_name in raw_paths:
            frame_id = os.path.splitext(raw_name)[0]
            raw_path = os.path.join(raw_dir, raw_name)
            with np.load(raw_path, allow_pickle=False) as raw:
                required = {
                    'shape_params', 'mhr_model_params', 'pred_cam_t',
                    'focal_length', 'width', 'height',
                }
                missing = sorted(required.difference(raw.files))
                if missing:
                    raise ValueError(f"{raw_path} is missing MHR fields: {missing}")

                shape = np.asarray(raw['shape_params'], dtype=np.float32).reshape(-1)
                model = np.asarray(raw['mhr_model_params'], dtype=np.float32).reshape(-1)
                translation = np.asarray(raw['pred_cam_t'], dtype=np.float32).reshape(-1)
                if shape.shape != (MHR_SHAPE_DIM,) or model.shape != (MHR_MODEL_PARAMS_DIM,) or translation.shape != (3,):
                    raise ValueError(
                        f"Unexpected MHR shapes in {raw_path}: shape={shape.shape}, "
                        f"model={model.shape}, cam_t={translation.shape}"
                    )
                if not all(np.isfinite(x).all() for x in (shape, model, translation)):
                    raise ValueError(f"Non-finite MHR parameters in {raw_path}")

                focal = float(raw['focal_length'])
                width = int(raw['width'])
                height = int(raw['height'])

            image_path = os.path.join(image_dir, f'{frame_id}.jpg')
            if not os.path.isfile(image_path):
                image_path = os.path.join(image_dir, f'{frame_id}.png')
            mask_path = os.path.join(mask_dir, f'{frame_id}.png')
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image for frame {frame_id} not found in {image_dir}")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask for frame {frame_id} not found: {mask_path}")

            frame_ids.append(frame_id)
            shape_params.append(shape)
            model_params.append(model)
            cam_t.append(translation)
            focal_lengths.append(focal)
            widths.append(width)
            heights.append(height)
            self._image_paths[frame_id] = image_path
            self._mask_paths[frame_id] = mask_path

        # The current 377 preparation stores a per-frame SAM-3D shape fit.
        # Keeping those values reproduces the saved MHR vertices exactly;
        # shape.npy remains available for consumers that want a fixed identity.
        shape_source = self.cfg.shape_source
        if shape_source not in ('raw', 'shape_npy'):
            raise ValueError(
                "dataset.shape_source must be 'raw' or 'shape_npy', "
                f"got {shape_source!r}"
            )
        if shape_source == 'shape_npy':
            shape_path = os.path.join(self.camera_dir, 'results', 'shape.npy')
            fixed_shape = np.asarray(np.load(shape_path), dtype=np.float32).reshape(-1)
            if fixed_shape.shape != (MHR_SHAPE_DIM,):
                raise ValueError(f"Unexpected shape.npy dimensions: {fixed_shape.shape}")
            shape_params = [fixed_shape.copy() for _ in frame_ids]

        # Real frame-holdout split: every cfg.test_every-th frame (by sorted
        # frame index) is held out for val/test, the rest is train. Previously
        # this method ignored self.split entirely, so train and val/test both
        # loaded the identical full frame set -- not a genuine split.
        test_every = int(self.cfg.test_every)
        is_test = [i % test_every == 0 for i in range(len(frame_ids))]
        keep = [not t for t in is_test] if self.split == 'train' else is_test

        def _select(values: list) -> list:
            return [v for v, k in zip(values, keep) if k]

        self.data_dir = self.camera_dir
        self.frame_ids = _select(frame_ids)
        self.parms = {
            'shape_params': torch.from_numpy(np.stack(_select(shape_params))),
            'model_params': torch.from_numpy(np.stack(_select(model_params))),
            'cam_t': torch.from_numpy(np.stack(_select(cam_t))),
            'focal_length': torch.tensor(_select(focal_lengths), dtype=torch.float32),
            'width': torch.tensor(_select(widths), dtype=torch.long),
            'height': torch.tensor(_select(heights), dtype=torch.long),
            'frame_ids': self.frame_ids,
        }
        self._image_paths = {f: self._image_paths[f] for f in self.frame_ids}
        self._mask_paths = {f: self._mask_paths[f] for f in self.frame_ids}
        self.is_keyframe = np.zeros(len(self.frame_ids), dtype=bool)

    def get_metadata(self) -> None:
        model_params_all = self.parms["model_params"].float()  # (N,204)
        shape_params_all = self.parms["shape_params"].float()  # (N,45)

        with torch.no_grad():
            big_shape = shape_params_all[0:1].to(self.device)
            big_model_params = build_big_pose_model_params(model_params_all[0]).unsqueeze(0).to(self.device)
            big_out = mhr_lbs.mhr_query(self.mhr_model, big_shape, big_model_params, device=self.device)
        big_pose_verts = self.body_model.rest_vertices().detach().cpu().numpy().astype(np.float32)
        big_pose_joint_pos = big_out["joint_pos"][0].cpu()  # (127,3)
        big_pose_joint_rotmat = big_out["joint_rotmat"][0].cpu()  # (127,3,3)

        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=float)
        coord_max = big_pose_verts.max(axis=0)
        coord_min = big_pose_verts.min(axis=0)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max = coord_max + padding
        coord_min = coord_min - padding

        center = big_pose_verts.mean(axis=0)
        centered = big_pose_verts - center
        cano_max = centered.max()
        cano_min = centered.min()
        norm_padding = (cano_max - cano_min) * self.cfg.canonical_norm_padding
        jtr_norm = big_pose_joint_pos.numpy() - center
        jtr_norm = (jtr_norm - cano_min + norm_padding) / (cano_max - cano_min) / self.cfg.jtr_norm_scale
        jtr_norm -= 0.5
        jtr_norm *= 2.0

        cano_mesh = trimesh.Trimesh(vertices=big_pose_verts.astype(np.float32), faces=self.faces)

        cano_data: MHRCanonicalMetadata = {
            'cano_verts': big_pose_verts,
            'skinning_weights': self.dense_skinning_weights.astype(np.float32),
            'faces': self.faces,
            'cano_mesh': cano_mesh,
            'joint_parents': self.joint_parents.numpy(),
            'big_pose_joint_pos': big_pose_joint_pos,
            'big_pose_joint_rotmat': big_pose_joint_rotmat,
            'jtr_norm': torch.from_numpy(jtr_norm).float(),
            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max.astype(np.float32), coord_min.astype(np.float32)),
            'body_model': self.body_model,
            'face_vertex_mask': self.body_model.face_region_mask(),
            'laplacian': self.body_model.laplacian(cano_mesh),
            'hand_only': self.hand_only,
            'hand_crop_padding': self.hand_crop_padding,
            'hand_joint_mask': self.body_model.hand_joint_mask(),
        }

        if self.split != 'train':
            self.metadata = cano_data
            return

        frame_dict = {frame: i for i, frame in enumerate(self.frame_ids)}
        self.metadata = MHRMetadata(
            **cano_data,
            cameras_extent=CAMERAS_EXTENT,
            frame_dict=frame_dict,
            # consumed by models/pose_correction/pose_correction.py's ShapeCorrection
            # (photometric-loss-driven optimization of the non-skeletal shape_params
            # only -- model_params/skeletal pose is frozen at its SAM3D-Body-direct-
            # or-interpolated value forever, see prepare_mhr_keyframe_track.py).
            mhr_model=self.mhr_model,
            mhr_model_path=self.cfg.mhr_model_path,
            model_params_all=model_params_all,
            shape_params_all=shape_params_all,
            cam_t_all=self.parms["cam_t"].float(),
            is_keyframe=self.is_keyframe,
        )

    def __len__(self) -> int:
        return len(self.frame_ids)

    def load_raw(self, idx: int) -> RawFrame:
        """CPU-only frame load (disk I/O + tensor bookkeeping, no CUDA). Safe to
        call from DataLoader worker subprocesses -- unlike building a Camera
        directly, which internally moves tensors to cfg.data_device (typically
        'cuda') and therefore must happen in the main process; see build_camera."""
        frame_id = self.frame_ids[idx]
        img_path = self._image_paths[frame_id]
        valid_path = self._mask_paths[frame_id]

        image_bgr = cv2.imread(img_path)
        valid_raw = cv2.imread(valid_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        if valid_raw is None:
            raise FileNotFoundError(f"Failed to load mask: {valid_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        valid = valid_raw != 0

        image_f = image.astype(np.float32)
        image_f[~valid] = 255. if self.white_bg else 0.
        image_f = image_f / 255.
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(valid).unsqueeze(0).float()

        width = int(self.parms["width"][idx])
        height = int(self.parms["height"][idx])
        focal = float(self.parms["focal_length"][idx])
        FoVx = focal2fov(focal, width)
        FoVy = focal2fov(focal, height)
        K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float32)

        pose_rot = self.pose_rot_all[idx]  # (127,9)
        bone_transforms = self.bone_transforms_all[idx]  # (127,4,4)

        return RawFrame(
            frame_id=frame_id,
            K=K,
            FoVx=FoVx, FoVy=FoVy,
            image=image_t,
            mask=mask_t,
            rots=pose_rot.float().unsqueeze(0),
            bone_transforms=bone_transforms.clone(),
        )

    def build_camera(self, raw: RawFrame) -> Camera:
        """Turn a load_raw() result into a Camera, including the CUDA moves --
        must run in the main process (see load_raw's docstring)."""
        return Camera(
            frame_id=raw['frame_id'],
            cam_id=0,
            K=raw['K'], R=np.eye(3, dtype=np.float32), T=np.zeros(3, dtype=np.float32),
            FoVx=raw['FoVx'], FoVy=raw['FoVy'],
            image=raw['image'],
            mask=raw['mask'],
            gt_alpha_mask=None,
            image_name=raw['frame_id'],
            data_device=self.cfg.data_device,
            rots=raw['rots'],
            Jtrs=self.metadata['jtr_norm'].unsqueeze(0),
            bone_transforms=raw['bone_transforms'].float(),
        )

    def getitem(self, idx: int) -> Camera:
        return self.build_camera(self.load_raw(idx))

    def __getitem__(self, idx: int) -> Camera:
        if self.preload:
            return self.cameras[idx]
        return self.getitem(idx)

    def raw_dataset(self) -> RawFrameDataset:
        """A CPU-only view of this dataset (load_raw instead of build_camera),
        suitable for a torch DataLoader with num_workers>0 -- see load_raw."""
        return RawFrameDataset(self)

    def load_depth_map(self, frame_id: str) -> torch.Tensor | None:
        """Load a GA-Avatar/Sapiens depth map as ``(1,H,W)``.

        The preprocessing script writes full-resolution ``float32`` arrays to
        ``sapiens_depth/<frame>.npy``.  Missing supervision remains optional,
        matching the SMPL-X dataset reader's behavior.
        """
        path = os.path.join(self._depth_dir, f"{frame_id}.npy")
        if not os.path.isfile(path):
            return None
        depth = np.load(path).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2-D depth map at {path}, got {depth.shape}")
        return torch.from_numpy(depth).unsqueeze(0).float()

    def readPointCloud(self) -> BasicPointCloud:
        if self.hand_only:
            xyz = np.asarray(self.metadata['cano_verts'], dtype=np.float32)
            return BasicPointCloud(
                points=xyz,
                colors=np.ones_like(xyz, dtype=np.float32),
                normals=np.zeros_like(xyz, dtype=np.float32),
            )
        ply_path = os.path.join(self.root_dir, 'cano_mhr.ply')
        try:
            pcd = fetchPly(ply_path)
        except Exception:
            verts = self.metadata['cano_verts']
            faces = self.faces
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            n_points = self.cfg.canonical_pointcloud_n_points
            xyz = mesh.sample(n_points)
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)
            pcd = fetchPly(ply_path)
        return pcd


class RawFrameDataset(Dataset[RawFrame]):
    """CPU-only Dataset wrapper for MHRNativeDataset, for use with a torch
    DataLoader(num_workers>0): each worker only ever touches load_raw (disk
    I/O + CPU tensor ops), never build_camera (which does the CUDA moves and
    so must stay in the main process)."""

    def __init__(self, dataset: MHRNativeDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> RawFrame:
        return self.dataset.load_raw(idx)

"""NeuMan (ExAvatar SMPL-X preprocessing) dataset reader for the GA-Avatar /
SMPL-X BodyModel path.

Reads a sequence directory in the layout produced by the ExAvatar project's
NeuMan preprocessing (see /mnt/ssd2/exavatar_neuman_combined/data/<seq>/):
  images/NNNNN.png, masks/NNNNN.png
  sparse/{cameras.txt,images.txt,points3D.txt}   -- standard COLMAP text format
  smpl_output_optimized.pkl                       -- joblib dict {1: {'pose': [ (72,) per frame ], 'betas': [...]}}
  alignments.npy                                  -- per-image {name: (4,3) array}; A[:3] is a
                                                       (possibly non-orthogonal, scale-carrying)
                                                       3x3 and A[3] a translation, mapping the raw
                                                       SMPL-posed mesh (SMPL's own local/root-
                                                       relative frame) into the same world frame
                                                       COLMAP's cameras are defined in:
                                                       world = posed_local @ A[:3] + A[3] (row-vector).
                                                       Verified empirically (not documented anywhere
                                                       in this preprocessing) by checking that a
                                                       frame's transformed mesh centroid lies within
                                                       ~12 degrees of that frame's camera viewing axis
                                                       -- see the implementation-plan chat log for the
                                                       numeric check. NOT the same coordinate frame as
                                                       smplx_optimized/meshes/*.ply, which live in
                                                       their own small-scale local frame and are only
                                                       used here as an initializer for CTO regularizers,
                                                       not for camera-space placement.
  smplx_optimized/joint_offset.json                -- (55,3) list; used as this dataset's delta_J_gt
                                                       (CTO's L_joint regularization target)
  smplx_optimized/face_offset.json                 -- (10475,3) list; only its nonzero-vertex support
                                                       is used, as the face-region-mask seed for
                                                       BodyModel.build_canonical_mesh's face_region_init
  {train,val,test}_split.txt                       -- newline-separated image basenames (e.g. 00000.png)

Per-frame body pose: `smpl_output_optimized.pkl`'s (72,) SMPL axis-angle pose
covers only SMPL's 24 joints (root + 23). SMPL-X's first 22 body joints
(root..right_wrist) share SMPL's joint topology/order exactly, so the first
22*3=66 values are copied directly into a 55-joint (165-dim) SMPL-X pose
vector; SMPL-X's remaining 33 joints (jaw, 2 eyes, 30 finger joints) have no
corresponding data in this preprocessing and are left at zero (rest pose) --
see `_assemble_smplx_pose`. This is a documented approximation: finger/jaw/
eye articulation is unavailable from this data source.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypedDict

import cv2
import joblib
import numpy as np
import numpy.typing as npt
import torch
import trimesh
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.body_models import get_body_model
from src.body_models.metadata import CanonicalMetadata, ModelMetadata
from src.dataset.training_interface import is_hand_dataset
from src.utils.dataset_utils import AABB, fetchPly, storePly
from src.utils.graphics_utils import BasicPointCloud, focal2fov

if TYPE_CHECKING:
    from src.scene.cameras import Camera
    from src.body_models.smplx_body_model import SMPLXBodyModel
else:
    # The dataset class uses Camera in its generic base annotation, but frame
    # preparation never constructs a Camera.  Avoid importing the full Scene
    # stack just to evaluate that annotation at runtime.
    Camera = Any
    SMPLXBodyModel = Any

N_SMPLX_JOINTS = 55
N_SMPL_BODY_JOINTS_SHARED = 22  # SMPL-X's root..right_wrist matches SMPL's joint 0..21 exactly


def _qvec2rotmat(qvec: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


class ColmapCamera(TypedDict):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _read_colmap_cameras(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                raise ValueError(f"Unsupported COLMAP camera model {model!r} in {path}")
            cameras[cam_id] = ColmapCamera(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


class ColmapImage(TypedDict):
    R: npt.NDArray[np.floating[Any]]  # world2cam rotation, TRANSPOSED (see getWorld2View2's convention)
    T: npt.NDArray[np.floating[Any]]  # world2cam translation
    camera_id: int


def _read_colmap_images(path: str) -> dict[str, ColmapImage]:
    images: dict[str, ColmapImage] = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qvec = np.array([float(p) for p in parts[1:5]])
        tvec = np.array([float(p) for p in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        R = np.transpose(_qvec2rotmat(qvec))
        images[name] = ColmapImage(R=R.astype(np.float32), T=tvec.astype(np.float32), camera_id=camera_id)
    return images


def _assemble_smplx_pose(smpl_pose72: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    pose = np.zeros(N_SMPLX_JOINTS * 3, dtype=np.float32)
    pose[: N_SMPL_BODY_JOINTS_SHARED * 3] = smpl_pose72[: N_SMPL_BODY_JOINTS_SHARED * 3]
    return pose


class RawFrame(TypedDict):
    frame_id: str
    K: npt.NDArray[np.floating[Any]]
    R: npt.NDArray[np.floating[Any]]
    T: npt.NDArray[np.floating[Any]]
    FoVx: float
    FoVy: float
    image: torch.Tensor
    mask: torch.Tensor
    rots: torch.Tensor
    bone_transforms: torch.Tensor
    align_matrix: torch.Tensor


class NeumanSMPLXDataset(Dataset[Camera]):
    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.root_dir: str = cfg.root_dir
        configured_hand_only = cfg.get("hand_only", None)
        self.hand_only = is_hand_dataset(
            self.root_dir,
            None if configured_hand_only is None else bool(configured_hand_only),
        )
        self.hand_crop_padding = float(cfg.get("hand_crop_padding", 0.25))
        self.white_bg: bool = cfg.get("white_background", False)
        self.data_device: str = cfg.get("data_device", "cuda")

        split_file = {
            "train": "train_split.txt",
            "val": "val_split.txt",
            "test": "test_split.txt",
            "predict": "test_split.txt",
        }[split]
        with open(os.path.join(self.root_dir, split_file)) as f:
            split_names = [l.strip() for l in f if l.strip()]
        self.frame_ids: list[str] = [os.path.splitext(n)[0] for n in sorted(split_names)]

        colmap_cameras = _read_colmap_cameras(os.path.join(self.root_dir, "sparse", "cameras.txt"))
        colmap_images = _read_colmap_images(os.path.join(self.root_dir, "sparse", "images.txt"))
        assert len(colmap_cameras) == 1, "NeumanSMPLXDataset assumes a single shared COLMAP camera model"
        self._colmap_cam = next(iter(colmap_cameras.values()))
        self._colmap_images = colmap_images

        pkl = joblib.load(os.path.join(self.root_dir, "smpl_output_optimized.pkl"))
        subject_key = next(iter(pkl.keys()))
        smpl_poses = pkl[subject_key]["pose"]  # list of (72,), one per image in sorted-filename order
        all_names = sorted(colmap_images.keys())
        self._pose72_by_name = dict(zip(all_names, smpl_poses))

        self._alignments = np.load(os.path.join(self.root_dir, "alignments.npy"), allow_pickle=True).item()

        self.body_model: SMPLXBodyModel = get_body_model(cfg.body_model)  # type: ignore[assignment]
        n_base_verts = self.body_model.model.v_template.shape[0]
        face_offset_path = os.path.join(self.root_dir, "smplx_optimized", "face_offset.json")
        face_region_init: npt.NDArray[np.bool_] | None = None
        if os.path.isfile(face_offset_path):
            import json

            fo = np.array(json.load(open(face_offset_path)), dtype=np.float32)
            if fo.shape[0] == n_base_verts:
                face_region_init = np.linalg.norm(fo, axis=1) > 1e-6
        self.body_model.build_canonical_mesh(
            n_subdivisions=int(cfg.body_model.get("n_subdivisions", 2)),
            face_region_init=face_region_init,
        )
        self.body_model.set_hand_only(self.hand_only)

        joint_offset_path = os.path.join(self.root_dir, "smplx_optimized", "joint_offset.json")
        self.joint_offset_gt: torch.Tensor
        if os.path.isfile(joint_offset_path):
            import json

            self.joint_offset_gt = torch.tensor(json.load(open(joint_offset_path)), dtype=torch.float32)
        else:
            self.joint_offset_gt = torch.zeros_like(self.body_model.joint_offset)

        self.metadata: CanonicalMetadata | ModelMetadata
        self.get_metadata()

        self._image_dir = os.path.join(self.root_dir, "images")
        self._mask_dir = os.path.join(
            self.root_dir, "hand_masks" if self.hand_only else "masks"
        )
        # Sapiens-precomputed geometric supervision (GA-Avatar's L_geo). Not
        # part of the ExAvatar preprocessing this dataset otherwise reads --
        # generated separately (see skills/ or the plan doc's data-prep
        # notes) via /mnt/ssd2/tavatar_data_prep/run_sapiens_normals.py +
        # sapiens_checkpoints/normal for normals; a Sapiens depth checkpoint
        # was not available on this machine as of writing, so depth_dir may
        # legitimately not exist -- load_depth_map/load_normal_map both
        # return None (not an error) when their file is missing, and
        # scripts/train.py's L_n/L_d terms silently contribute 0 for that
        # frame in that case.
        self._normal_dir = os.path.join(self.root_dir, "sapiens_normal")
        self._depth_dir = os.path.join(self.root_dir, "sapiens_depth")

        self.preload: bool = cfg.get("preload", True)
        self.cameras: list[Camera] = []
        if self.preload:
            self.cameras = [self.getitem(i) for i in range(len(self))]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def align_matrix(self, frame_id: str) -> torch.Tensor:
        """(4,4) world-alignment matrix for `frame_id`, in the column-vector
        (M @ [v;1]) convention -- see module docstring for the row-vector
        source form this is transposed from."""
        A = self._alignments[f"{frame_id}.png"]
        Amat, trans = A[:3], A[3]
        M = torch.eye(4)
        M[:3, :3] = torch.from_numpy(Amat.T.astype(np.float32))
        M[:3, 3] = torch.from_numpy(trans.astype(np.float32))
        return M

    def assembled_pose(self, frame_id: str) -> torch.Tensor:
        """(1, 55*3) SMPL-X axis-angle pose for `frame_id` -- see
        `_assemble_smplx_pose`."""
        pose72 = self._pose72_by_name[f"{frame_id}.png"]
        return torch.from_numpy(_assemble_smplx_pose(pose72)).unsqueeze(0)

    def _world_bone_transforms(self, frame_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(joint_pos_world (55,3), bone_transforms_world (55,4,4)) for one
        frame, computed with the body model's parameters AT CALL TIME --
        composing the SMPL-X FK's local (root-relative) bone transforms with
        this frame's alignments.npy world-placement (see module docstring).
        Values stored on a preloaded Camera (camera.bone_transforms) are a
        SNAPSHOT taken once at dataset-init time (see load_raw) and go stale
        as soon as CTO parameters (joint_offset, beta) change during
        training -- live-CTO-aware consumers (e.g. VertexLBS) must call this
        again each step using `camera.rots` + `camera.align_matrix` instead
        of trusting the snapshot; see Camera.align_matrix's docstring."""
        # scripts/train.py's DataLoader keeps calling load_raw (hence this)
        # throughout training, long after body_model has been moved to cuda
        # by GaussianConverter(...).cuda() in Scene.__init__ -- match
        # whatever device it's currently on rather than assuming CPU.
        device = self.body_model.beta.device
        pose_smplx = self.assembled_pose(frame_id).to(device)
        joint_pos, _joint_rotmat, bone_transforms_local = self.body_model.pose_transforms(pose_smplx)
        joint_pos = joint_pos[0]
        bone_transforms_local = bone_transforms_local[0]  # (55,4,4)

        M = self.align_matrix(frame_id).to(device)
        bone_transforms_world = torch.matmul(M.unsqueeze(0), bone_transforms_local)  # (55,4,4)
        homo = torch.cat([joint_pos, torch.ones(joint_pos.shape[0], 1, device=device)], dim=1)
        joint_pos_world = torch.einsum("jab,jb->ja", M.unsqueeze(0).expand(joint_pos.shape[0], -1, -1), homo)[:, :3]
        return joint_pos_world, bone_transforms_world

    def get_metadata(self) -> None:
        cano_verts = self.body_model.rest_vertices().detach().cpu().numpy().astype(np.float32)
        faces = self.body_model.faces()
        skinning_weights = self.body_model.skinning_weights().detach().cpu().numpy().astype(np.float32)
        joint_parents = self.body_model.joint_parents()

        # "Big pose" for AABB/jtr_norm purposes: SMPL-X's own zero (rest) pose,
        # which is already a reasonable non-self-intersecting A-pose -- unlike
        # MHR, no de-crossing correction is needed here (see
        # src/body_models/mhr_utils.py's build_big_pose_model_params for why
        # MHR needs one and this doesn't).
        zero_pose = torch.zeros(1, N_SMPLX_JOINTS * 3)
        with torch.no_grad():
            big_pose_joint_pos, big_pose_joint_rotmat, _ = self.body_model.pose_transforms(zero_pose)
        big_pose_joint_pos = big_pose_joint_pos[0]
        big_pose_joint_rotmat = big_pose_joint_rotmat[0]

        padding_ratio = np.array(self.cfg.get("padding", 0.1), dtype=float)
        coord_max = cano_verts.max(axis=0)
        coord_min = cano_verts.min(axis=0)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max = coord_max + padding
        coord_min = coord_min - padding

        center = cano_verts.mean(axis=0)
        centered = cano_verts - center
        cano_max = centered.max()
        cano_min = centered.min()
        norm_padding = (cano_max - cano_min) * self.cfg.get("canonical_norm_padding", 0.05)
        jtr_norm = big_pose_joint_pos.numpy() - center
        jtr_norm = (jtr_norm - cano_min + norm_padding) / (cano_max - cano_min) / self.cfg.get("jtr_norm_scale", 1.1)
        jtr_norm -= 0.5
        jtr_norm *= 2.0

        cano_mesh = trimesh.Trimesh(vertices=cano_verts, faces=faces, process=False)

        cano_data: CanonicalMetadata = {
            "cano_verts": cano_verts,
            "skinning_weights": skinning_weights,
            "faces": faces,
            "cano_mesh": cano_mesh,
            "joint_parents": joint_parents,
            "big_pose_joint_pos": big_pose_joint_pos,
            "big_pose_joint_rotmat": big_pose_joint_rotmat,
            "jtr_norm": torch.from_numpy(jtr_norm).float(),
            "coord_min": coord_min,
            "coord_max": coord_max,
            "aabb": AABB(coord_max.astype(np.float32), coord_min.astype(np.float32)),
            "body_model": self.body_model,
            "face_vertex_mask": self.body_model.face_region_mask(),
            "laplacian": self.body_model.laplacian(cano_mesh),
            "hand_only": self.hand_only,
            "hand_crop_padding": self.hand_crop_padding,
            "hand_joint_mask": self.body_model.hand_joint_mask(),
        }

        if self.split != "train":
            self.metadata = cano_data
            return

        frame_dict = {frame: i for i, frame in enumerate(self.frame_ids)}
        self.metadata = ModelMetadata(**cano_data, cameras_extent=1.0, frame_dict=frame_dict)

    def load_raw(self, idx: int) -> RawFrame:
        frame_id = self.frame_ids[idx]
        image_bgr = cv2.imread(os.path.join(self._image_dir, f"{frame_id}.png"))
        mask_raw = cv2.imread(os.path.join(self._mask_dir, f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask_raw is None:
            raise FileNotFoundError(f"Missing image/mask for frame {frame_id} in {self.root_dir}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cam = self._colmap_cam
        colmap_image = self._colmap_images[f"{frame_id}.png"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        width, height = cam["width"], cam["height"]

        # downsample: this preprocessing's images (1276x717) + a full
        # 167K-vertex canonical mesh need ~20GB of diff_gaussian_rasterization
        # scratch buffers to rasterize at full resolution -- more than fits
        # on hardware smaller than the paper's RTX 4090 (24GB). Not a paper
        # deviation in Gaussian count/topology, only in rendered resolution.
        downsample = float(self.cfg.get("downsample", 1))
        if downsample != 1:
            new_w, new_h = int(round(width / downsample)), int(round(height / downsample))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_raw = cv2.resize(mask_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
            fx, fy, cx, cy = fx / downsample, fy / downsample, cx / downsample, cy / downsample
            width, height = new_w, new_h

        valid = mask_raw > 127
        image_f = image.astype(np.float32)
        image_f[~valid] = 255.0 if self.white_bg else 0.0
        image_f = image_f / 255.0
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(valid).unsqueeze(0).float()

        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        with torch.no_grad():
            _joint_pos_world, bone_transforms_world = self._world_bone_transforms(frame_id)
        rots = self.assembled_pose(frame_id)  # (1,165): raw pose-conditioning input for GeoNet/RGBNet refine MLPs

        return RawFrame(
            frame_id=frame_id,
            K=K,
            R=colmap_image["R"],
            T=colmap_image["T"],
            FoVx=FoVx,
            FoVy=FoVy,
            image=image_t,
            mask=mask_t,
            rots=rots,
            bone_transforms=bone_transforms_world.detach(),
            align_matrix=self.align_matrix(frame_id),
        )

    def build_camera(self, raw: RawFrame) -> Camera:
        from src.scene.cameras import Camera

        return Camera(
            frame_id=raw["frame_id"],
            cam_id=0,
            K=raw["K"], R=raw["R"], T=raw["T"],
            FoVx=raw["FoVx"], FoVy=raw["FoVy"],
            image=raw["image"],
            mask=raw["mask"],
            gt_alpha_mask=None,
            image_name=raw["frame_id"],
            data_device=self.data_device,
            rots=raw["rots"],
            align_matrix=raw["align_matrix"],
            Jtrs=self.metadata["jtr_norm"].unsqueeze(0),
            bone_transforms=raw["bone_transforms"].unsqueeze(0).float(),
        )

    def getitem(self, idx: int) -> Camera:
        return self.build_camera(self.load_raw(idx))

    def __getitem__(self, idx: int) -> Camera:
        if self.preload:
            return self.cameras[idx]
        return self.getitem(idx)

    def raw_dataset(self) -> RawFrameDataset:
        """CPU-only view for scripts/train.py's DataLoader (matches
        MHRNativeDataset.raw_dataset's contract exactly)."""
        return RawFrameDataset(self)

    def readPointCloud(self) -> BasicPointCloud:
        if self.hand_only:
            xyz = np.asarray(self.metadata["cano_verts"], dtype=np.float32)
            return BasicPointCloud(
                points=xyz,
                colors=np.ones_like(xyz, dtype=np.float32),
                normals=np.zeros_like(xyz, dtype=np.float32),
            )
        ply_path = os.path.join(self.root_dir, "cano_smplx.ply")
        try:
            pcd = fetchPly(ply_path)
        except Exception:
            verts = self.metadata["cano_verts"]
            rgb = np.ones_like(verts) * 255
            storePly(ply_path, verts, rgb)
            pcd = fetchPly(ply_path)
        return pcd

    def _target_hw(self) -> tuple[int, int]:
        """(H,W) that load_raw actually renders at, after `downsample` --
        Sapiens maps are precomputed at the source (1276x717) resolution, so
        they must be resized to match `depth_render`/`normal_render`'s shape
        or the loss comparison shape-mismatches."""
        cam = self._colmap_cam
        width, height = cam["width"], cam["height"]
        downsample = float(self.cfg.get("downsample", 1))
        if downsample == 1:
            return height, width
        return int(round(height / downsample)), int(round(width / downsample))

    def load_normal_map(self, frame_id: str) -> torch.Tensor | None:
        """(3,H,W) normal map in [-1,1] for `frame_id`, or None if this
        sequence hasn't had Sapiens normals generated yet -- see the
        `_normal_dir` note in __init__."""
        path = os.path.join(self._normal_dir, f"{frame_id}.png")
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        target_h, target_w = self._target_hw()
        if (img.shape[0], img.shape[1]) != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        normal = img / 127.5 - 1.0  # standard normal-map PNG encoding: [0,255] -> [-1,1]
        normal_t = torch.from_numpy(normal).permute(2, 0, 1).float()
        return normal_t / normal_t.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def load_depth_map(self, frame_id: str) -> torch.Tensor | None:
        """(1,H,W) relative depth map for `frame_id`, or None if this
        sequence hasn't had Sapiens depth generated -- see the `_depth_dir`
        note in __init__. `depth_loss_local`/`depth_loss_global` normalize
        scale/offset internally, so this map's arbitrary Sapiens units don't
        need to be metric."""
        path = os.path.join(self._depth_dir, f"{frame_id}.npy")
        if not os.path.isfile(path):
            return None
        depth = np.load(path).astype(np.float32)
        target_h, target_w = self._target_hw()
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(depth).unsqueeze(0).float()


class RawFrameDataset(Dataset[RawFrame]):
    """CPU-only Dataset wrapper for NeumanSMPLXDataset, for use with a torch
    DataLoader(num_workers>0) -- mirrors MHRNativeDataset's RawFrameDataset
    exactly: each worker only ever touches load_raw (disk I/O + CPU tensor
    ops), never build_camera (which does the CUDA moves)."""

    def __init__(self, dataset: NeumanSMPLXDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> RawFrame:
        return self.dataset.load_raw(idx)

"""SMPL-X `BodyModel` backend, implementing GA-Avatar's Canonical Template
Optimization (CTO) on top of the `smplx` package.

Pipeline (matches the paper's Sec. III-B/III-C):
  1. Pytorch3D-subdivide the STANDARD (beta=0) SMPL-X template once to build a
     high-resolution canonical topology (~167K verts / ~335K faces). The
     base template's per-vertex shape-blendshape directions (`shapedirs`) and
     LBS skinning weights (`lbs_weights`) are subdivided (edge-midpoint
     averaged) alongside the mesh itself, via pytorch3d.ops.SubdivideMeshes'
     `feats` argument, so every upsampled vertex gets its own (interpolated)
     blendshape direction and skinning-weight row.
  2. CTO's learnable {beta, joint_offset, face_offset} are then applied on
     top of this fixed high-res topology: rest_vertices() = subdivided
     v_template + subdivided_shapedirs @ beta + face_offset (face_offset is
     itself a per-upsampled-vertex free parameter, non-zero only in the face
     region). joint_offset (55,3) is resolution-independent and is added to
     the *base*-mesh joint regression (joints are a skeletal quantity, not a
     surface-resolution one).
  3. pose_transforms() evaluates joint regression on the base
     (unsubdivided) shaped template -- consistent with standard SMPL-X, which
     defines J_regressor only for the 10475-vertex topology -- then applies
     the resulting per-joint transforms to the canonical (subdivided)
     vertices using their propagated skinning weights (done by callers via
     `skinning_weights()`, not inside `pose_transforms()`, since vertex LBS is
     shared by all backends in `BodyModel.lbs`.

Per-frame body pose theta is NEVER a parameter of this class -- CTO only
covers {beta, joint_offset, face_offset}, matching the paper's statement that
pose is held fixed throughout training (see cto_parameters()).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import smplx
import smplx.lbs as smplx_lbs
import torch
import torch.nn as nn
import trimesh
from omegaconf import DictConfig
from src.body_models.base import BODY_PARTS, BodyModel


class SMPLXBodyModel(BodyModel, nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg

        self.model = smplx.SMPLX(
            model_path=cfg.model_path,
            gender=cfg.get("gender", "neutral"),
            num_betas=int(cfg.get("num_betas", 10)),
            use_pca=False,
            flat_hand_mean=True,
            ext="npz",
        )
        self.model.requires_grad_(False)

        n_base_verts = self.model.v_template.shape[0]
        n_joints = self.model.parents.shape[0]
        num_betas = self.model.num_betas

        init_betas = cfg.get("init_betas", None)
        beta0 = (
            torch.tensor(init_betas, dtype=torch.float32).reshape(1, -1)
            if init_betas is not None
            else torch.zeros(1, num_betas)
        )
        assert beta0.shape == (1, num_betas), (
            f"init_betas has {beta0.shape[1]} entries, expected {num_betas} "
            "(= body_model.num_betas)"
        )
        self.beta = nn.Parameter(beta0)
        self.joint_offset = nn.Parameter(torch.zeros(n_joints, 3))
        # face_offset lives at the (post-subdivision) canonical-mesh
        # resolution, so it can't be allocated until build_canonical_mesh()
        # has run; see that method.
        self.face_offset: nn.Parameter | None = None

        self._built = False
        # Populated by build_canonical_mesh():
        self._cano_faces: npt.NDArray[np.integer[Any]]
        self._subdiv_shapedirs: torch.Tensor  # (V_hi, 3, num_betas)
        self._subdiv_lbs_weights: torch.Tensor  # (V_hi, n_joints)
        self._subdiv_v_template: torch.Tensor  # (V_hi, 3)
        self._face_region_mask: npt.NDArray[np.bool_]
        self._hand_joint_grad_hook: torch.utils.hooks.RemovableHandle | None = None

    def build_canonical_mesh(
        self,
        n_subdivisions: int = 2,
        face_region_init: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        """One-time (called by the dataset's get_metadata()) construction of
        the high-resolution canonical topology. `face_region_init`, if given,
        is a (10475,) boolean mask over the BASE SMPL-X mesh (e.g. derived
        from an external per-vertex face-offset fit's nonzero support) that
        gets propagated onto the upsampled mesh the same way shapedirs/
        lbs_weights are."""
        device = self.model.v_template.device
        base_verts = self.model.v_template.detach()
        base_faces = torch.as_tensor(self.model.faces.astype(np.int64), device=device)

        num_betas = self.model.num_betas
        n_joints = self.model.parents.shape[0]
        feats = torch.cat(
            [
                self.model.shapedirs.detach().reshape(base_verts.shape[0], -1),  # (V,3*num_betas)
                self.model.lbs_weights.detach(),  # (V,n_joints)
                (
                    torch.from_numpy(face_region_init.astype(np.float32)).reshape(-1, 1)
                    if face_region_init is not None
                    else torch.zeros(base_verts.shape[0], 1)
                ).to(device),
            ],
            dim=1,
        )

        verts, faces, feats_out = base_verts, base_faces, feats
        for _ in range(n_subdivisions):
            from pytorch3d.ops import SubdivideMeshes
            from pytorch3d.structures import Meshes

            meshes = Meshes(verts=[verts], faces=[faces])
            subdivider = SubdivideMeshes()
            new_meshes, feats_out = subdivider(meshes, feats_out)
            verts = new_meshes.verts_list()[0]
            faces = new_meshes.faces_list()[0]

        # Registered as buffers (not plain attributes) so `.cuda()`/`.to(...)`
        # calls on this nn.Module -- which only otherwise touch nn.Parameters
        # and submodules -- also move these along with the rest of the model.
        self.register_buffer("_subdiv_v_template", verts.contiguous())
        self._cano_faces = faces.cpu().numpy().astype(np.int64)
        self.register_buffer(
            "_subdiv_shapedirs", feats_out[:, : 3 * num_betas].reshape(-1, 3, num_betas).contiguous()
        )
        self.register_buffer(
            "_subdiv_lbs_weights", feats_out[:, 3 * num_betas : 3 * num_betas + n_joints].contiguous()
        )
        face_mask_soft = feats_out[:, -1]
        self._face_region_mask = (face_mask_soft > 0.5).cpu().numpy()

        self.face_offset = nn.Parameter(torch.zeros(verts.shape[0], 3))
        self._built = True

    def _assert_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "SMPLXBodyModel.build_canonical_mesh() must be called once "
                "(by the dataset's get_metadata()) before rest_vertices()/"
                "faces()/skinning_weights()/face_region_mask()/lbs() are used."
            )

    # --- BodyModel interface -------------------------------------------------

    def rest_vertices(self) -> torch.Tensor:
        self._assert_built()
        assert self.face_offset is not None
        shape_offset = torch.einsum("vcb,b->vc", self._subdiv_shapedirs, self.beta[0])
        vertices = self._subdiv_v_template + shape_offset + self.face_offset
        return vertices[self.hand_vertex_mask().to(vertices.device)] if self.hand_only else vertices

    def faces(self) -> npt.NDArray[np.integer[Any]]:
        self._assert_built()
        return self.select_vertex_faces(self._cano_faces, self.hand_vertex_mask()) if self.hand_only else self._cano_faces

    def skinning_weights(self) -> torch.Tensor:
        self._assert_built()
        return self._subdiv_lbs_weights[self.hand_vertex_mask()] if self.hand_only else self._subdiv_lbs_weights

    def joint_parents(self) -> npt.NDArray[np.integer[Any]]:
        return np.asarray(self.model.parents.detach().cpu().numpy(), dtype=np.int64)

    def face_region_mask(self) -> npt.NDArray[np.bool_]:
        self._assert_built()
        return self._face_region_mask[self.hand_vertex_mask().cpu().numpy()] if self.hand_only else self._face_region_mask

    # SMPL-X: body wrists are joints 20/21 and articulated fingers are joints
    # 25..54 (22..24 are jaw/eyes), split evenly left/right (15 finger joints
    # each) per the standard SMPL-X joint ordering.
    _LEFT_HAND_JOINTS = (20, *range(25, 40))
    _RIGHT_HAND_JOINTS = (21, *range(40, 55))

    def _hand_joints(self) -> tuple[int, ...]:
        if self.hand_side == "left":
            return self._LEFT_HAND_JOINTS
        if self.hand_side == "right":
            return self._RIGHT_HAND_JOINTS
        return (*self._LEFT_HAND_JOINTS, *self._RIGHT_HAND_JOINTS)

    def hand_vertex_mask(self) -> torch.Tensor:
        self._assert_built()
        return self._subdiv_lbs_weights[:, self._hand_joints()].sum(dim=1) >= 0.5

    def hand_pose_parameter_mask(self) -> torch.Tensor:
        n_joints = int(self.model.parents.shape[0])
        mask = torch.zeros(n_joints, 3, dtype=torch.bool)
        mask[list(self._hand_joints())] = True
        return mask.reshape(-1)

    def hand_joint_mask(self) -> torch.Tensor:
        return self.hand_pose_parameter_mask().reshape(-1, 3).any(dim=1)

    # Standard SMPL-X joint ordering (55 joints): 0 pelvis, 1/2 hips, 3/6/9
    # spine1-3, 4/5 knees, 7/8 ankles, 10/11 feet, 12 neck, 13/14 collars, 15
    # head, 16/17 shoulders, 18/19 elbows, 20/21 wrists (= _LEFT/RIGHT_HAND_
    # JOINTS' wrist entries), 22 jaw, 23/24 eyes, 25..54 finger joints (also
    # in _LEFT/RIGHT_HAND_JOINTS). Arms stop at the elbow so they don't
    # overlap the hand joints above; the neck stays with the torso so "head"
    # is exactly the head/jaw/eyes group.
    _LEFT_ARM_JOINTS = (16, 18)
    _RIGHT_ARM_JOINTS = (17, 19)
    _LEFT_LEG_JOINTS = (1, 4, 7, 10)
    _RIGHT_LEG_JOINTS = (2, 5, 8, 11)
    _HEAD_JOINTS = (15, 22, 23, 24)
    _TORSO_JOINTS = (0, 3, 6, 9, 12, 13, 14)

    def body_part_vertex_mask(self, part: str) -> torch.Tensor:
        self._assert_built()
        if part == "left_hand":
            joints = self._LEFT_HAND_JOINTS
        elif part == "right_hand":
            joints = self._RIGHT_HAND_JOINTS
        elif part == "left_arm":
            joints = self._LEFT_ARM_JOINTS
        elif part == "right_arm":
            joints = self._RIGHT_ARM_JOINTS
        elif part == "left_leg":
            joints = self._LEFT_LEG_JOINTS
        elif part == "right_leg":
            joints = self._RIGHT_LEG_JOINTS
        elif part == "head":
            joints = self._HEAD_JOINTS
        elif part == "torso":
            joints = self._TORSO_JOINTS
        else:
            raise ValueError(f"Unknown body part {part!r}, expected one of {BODY_PARTS}")
        return self._subdiv_lbs_weights[:, list(joints)].sum(dim=1) >= 0.5

    def set_hand_only(self, enabled: bool = True, side: str | None = None) -> None:
        super().set_hand_only(enabled, side)
        if enabled and self._hand_joint_grad_hook is None:
            joint_mask = self.hand_joint_mask()
            self._hand_joint_grad_hook = self.joint_offset.register_hook(
                lambda grad: grad * joint_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
            )

    def pose_transforms(self, pose_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pose_params: (B, n_joints*3) full SMPL-X axis-angle pose (root
        (global_orient) + 54 body/hand/jaw/eye joints), one row per frame --
        assembled by the dataset from whatever per-frame pose source it has
        (see src/dataset/neuman_smplx.py). Joint regression runs on the BASE
        (10475-vertex) shaped template, per SMPL-X convention; joint_offset
        is added post-regression."""
        n_joints = self.model.parents.shape[0]
        if pose_params.ndim == 1:
            pose_params = pose_params.unsqueeze(0)
        if pose_params.ndim != 2 or pose_params.shape[-1] != n_joints * 3:
            raise ValueError(
                "SMPL-X pose parameters must have shape "
                f"({n_joints * 3},) or (B,{n_joints * 3}), got {tuple(pose_params.shape)}"
            )
        base_v_shaped = self.model.v_template + smplx_lbs.blend_shapes(self.beta, self.model.shapedirs)
        J = smplx_lbs.vertices2joints(self.model.J_regressor, base_v_shaped) + self.joint_offset.unsqueeze(0)

        batch_size = pose_params.shape[0]
        J = J.expand(batch_size, -1, -1)
        rot_mats = smplx_lbs.batch_rodrigues(pose_params.reshape(-1, 3)).view(batch_size, n_joints, 3, 3)
        joint_pos, bone_transforms = smplx_lbs.batch_rigid_transform(rot_mats, J, self.model.parents)
        joint_rotmat = bone_transforms[..., :3, :3]
        return joint_pos, joint_rotmat, bone_transforms

    def subdivide(
        self, mesh: trimesh.Trimesh, n_iters: int
    ) -> tuple[trimesh.Trimesh, npt.NDArray[np.integer[Any]]]:
        """Generic single-mesh subdivision (not the CTO canonical-mesh path,
        which is build_canonical_mesh() above -- this is for ad-hoc callers,
        e.g. debug tooling, that want to subdivide an arbitrary posed mesh
        without needing skinning-weight/face-mask propagation). `parent_idx`
        is derived by nearest-neighbor lookup against the ORIGINAL vertex
        positions after subdivision (not by averaging an index feature
        through SubdivideMeshes, which would blend unrelated indices for new
        midpoint vertices and produce nonsense)."""
        from pytorch3d.ops import knn_points

        device = self.model.v_template.device
        orig_verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        verts = orig_verts
        faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
        for _ in range(n_iters):
            meshes = Meshes(verts=[verts], faces=[faces])
            new_meshes = SubdivideMeshes()(meshes)
            verts = new_meshes.verts_list()[0]
            faces = new_meshes.faces_list()[0]
        _, idx, _ = knn_points(verts[None], orig_verts[None], K=1)
        parent_idx = idx.view(-1).cpu().numpy()
        new_mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy(), process=False)
        return new_mesh, parent_idx

    def cto_parameters(self) -> dict[str, nn.Parameter]:
        self._assert_built()
        assert self.face_offset is not None
        return {
            "beta": self.beta,
            "joint_offset": self.joint_offset,
            "face_offset": self.face_offset,
        }

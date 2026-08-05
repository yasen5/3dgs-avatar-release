"""Backend-agnostic canonical/model metadata TypedDicts.

This is a pure rename of what used to live only as `MHRCanonicalMetadata`/
`MHRMetadata` in src/dataset/mhr_native.py -- field names and semantics are
unchanged, so every existing MHR call site keeps working unmodified via the
aliases re-exported from mhr_native.py. The generalization only adds three
new optional keys GA-Avatar needs (`body_model`, `face_vertex_mask`,
`laplacian`); MHR-only fields (`mhr_model`, `mhr_model_path`, ...) stay put
since they're only ever read by MHR-specific consumers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict
from typing_extensions import NotRequired

import numpy as np
import numpy.typing as npt
import torch
import trimesh

from src.utils.dataset_utils import AABB

if TYPE_CHECKING:
    from src.body_models.base import BodyModel


class CanonicalMetadata(TypedDict):
    """Fields present regardless of split (train vs. val/test/predict)."""
    cano_verts: npt.NDArray[np.floating[Any]]
    skinning_weights: npt.NDArray[np.floating[Any]]
    faces: npt.NDArray[np.integer[Any]]
    cano_mesh: trimesh.Trimesh
    joint_parents: npt.NDArray[np.integer[Any]]
    big_pose_joint_pos: torch.Tensor
    big_pose_joint_rotmat: torch.Tensor
    jtr_norm: torch.Tensor
    coord_min: npt.NDArray[np.floating[Any]]
    coord_max: npt.NDArray[np.floating[Any]]
    aabb: AABB

    # GA-Avatar / BodyModel-abstraction additions (optional: only populated by
    # backends/datasets that use them, e.g. the SMPL-X + GA-Avatar path).
    body_model: NotRequired["BodyModel"]
    face_vertex_mask: NotRequired[npt.NDArray[np.bool_]]
    laplacian: NotRequired[Any]  # scipy.sparse matrix over cano_mesh's vertices


class ModelMetadata(CanonicalMetadata):
    """Metadata for the (always-loaded) 'train' split, the only split any
    model consumer reads (see `Scene.metadata`)."""
    cameras_extent: float
    frame_dict: dict[str, int]

    # MHR-specific fields (present only for the MHR backend; MHR-only
    # consumers such as models/pose_correction/pose_correction.py's
    # ShapeCorrection read these directly).
    mhr_model: NotRequired[torch.jit.ScriptModule]
    mhr_model_path: NotRequired[str]
    model_params_all: NotRequired[torch.Tensor]
    shape_params_all: NotRequired[torch.Tensor]
    cam_t_all: NotRequired[torch.Tensor]
    is_keyframe: NotRequired[npt.NDArray[np.bool_]]

"""MHR implementation of the common :mod:`src.body_models` contract.

MHR exposes a complete posed mesh from its TorchScript rig, but the rest of
the codebase needs the same pieces that SMPL-X exposes: a canonical mesh,
faces, dense weights, a joint hierarchy, and transforms that can be fed to a
shared LBS routine.  This adapter queries MHR at the canonical and requested
poses and takes the relative transform between the two joint states.  The
actual vertex blending is then performed by ``BodyModel.lbs`` for both rigs.

MHR's public checkpoint is in centimetres and uses the opposite signs for its
second and third coordinates.  All tensors exposed here use the repository's
metre, axis-flipped convention, matching ``mhr_lbs.mhr_query`` and the
existing MHR dataset.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import trimesh
from omegaconf import DictConfig
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from src.body_models import mhr_lbs
from src.body_models.base import BodyModel
from src.body_models.mhr_lbs import MHRQueryOutput
from src.constants import MHR_HAND_DIMS


NUM_MHR_SHAPE_PARAMS = 45
NUM_MHR_MODEL_PARAMS = 204
NUM_MHR_EXPR_PARAMS = 72


def _tensor_config_value(
    cfg: DictConfig,
    name: str,
    size: int,
    *,
    aliases: tuple[str, ...] = (),
) -> torch.Tensor:
    value: Any = cfg.get(name, None)
    if value is None:
        for alias in aliases:
            value = cfg.get(alias, None)
            if value is not None:
                break
    if value is None:
        return torch.zeros(1, size, dtype=torch.float32)
    result = torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)
    if result.shape != (1, size):
        raise ValueError(f"{name} must contain {size} values, got {tuple(result.shape)}")
    return result


class MHRBodyModel(BodyModel, nn.Module):
    """BodyModel adapter for the Meta Momentum Human Rig.

    ``pose_params`` are MHR's 204 model parameters, with shape ``(204,)`` or
    ``(B,204)``.  The shape and expression vectors are initialized from the
    config and remain fixed unless a caller explicitly optimizes the exposed
    ``shape_params`` parameter.  ``canonical_model_params`` selects the pose
    in which ``rest_vertices()`` is expressed; zero is the rig's true rest
    pose, while an application may initialize it to a big/star pose before
    handing the model to the rest of the pipeline.
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        model: torch.jit.ScriptModule | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.cfg = cfg
        model_path = cfg.get("model_path", cfg.get("mhr_model_path", None))
        if model_path is None:
            raise ValueError("MHR body model requires model_path (or mhr_model_path)")
        self.model_path = str(model_path)
        device = str(cfg.get("device", cfg.get("mhr_device", "cpu")))
        self._mhr_device = torch.device(device)

        self.model = model if model is not None else mhr_lbs.load_mhr(self.model_path, device=device)
        state_dict = state_dict if state_dict is not None else mhr_lbs.mhr_state_dict(self.model_path)

        self._faces: torch.Tensor
        self._skinning_weights: torch.Tensor
        self._joint_parents: torch.Tensor
        self._neutral_rest_vertices: torch.Tensor
        self._canonical_model_params: torch.Tensor
        self._expr_params: torch.Tensor
        self._face_region_mask: torch.Tensor
        self.register_buffer("_faces", mhr_lbs.faces(state_dict).long().cpu())
        self.register_buffer(
            "_skinning_weights",
            mhr_lbs.dense_skinning_weights(state_dict).float().cpu(),
        )
        self.register_buffer("_joint_parents", mhr_lbs.joint_parents(state_dict).long().cpu())

        # The checkpoint stores neutral rest vertices in centimetres, before
        # the same coordinate conversion applied by mhr_lbs.mhr_query.  Keep a
        # neutral copy for diagnostics; shaped canonical vertices are obtained
        # from the rig in rest_vertices().
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        neutral_rest = mhr_lbs.rest_vertices(state_dict).float() / 100.0
        self.register_buffer("_neutral_rest_vertices", neutral_rest @ flip.T)

        shape = _tensor_config_value(cfg, "shape_params", NUM_MHR_SHAPE_PARAMS)
        self.shape_params = nn.Parameter(
            shape,
            requires_grad=bool(cfg.get("learn_shape", False)),
        )
        self.register_buffer(
            "_canonical_model_params",
            _tensor_config_value(
                cfg,
                "canonical_model_params",
                NUM_MHR_MODEL_PARAMS,
                aliases=("reference_model_params",),
            )[0],
        )
        self.register_buffer(
            "_expr_params",
            _tensor_config_value(cfg, "expr_params", NUM_MHR_EXPR_PARAMS)[0],
        )

        face_mask_value = cfg.get("face_vertex_mask", None)
        if face_mask_value is None:
            face_mask = torch.zeros(int(self._neutral_rest_vertices.shape[0]), dtype=torch.bool)
        else:
            face_mask = torch.as_tensor(face_mask_value, dtype=torch.bool).reshape(-1)
            if face_mask.shape != (self._neutral_rest_vertices.shape[0],):
                raise ValueError(
                    "face_vertex_mask must have one entry per MHR vertex, got "
                    f"{tuple(face_mask.shape)}"
                )
        self.register_buffer("_face_region_mask", face_mask)

    @property
    def num_joints(self) -> int:
        return int(self._joint_parents.shape[0])

    @property
    def num_verts(self) -> int:
        return int(self._neutral_rest_vertices.shape[0])

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            try:
                return next(self.model.buffers()).device
            except StopIteration:
                return self._mhr_device

    def _query(self, model_params: torch.Tensor) -> MHRQueryOutput:
        if model_params.ndim == 1:
            model_params = model_params.unsqueeze(0)
        if model_params.ndim != 2 or model_params.shape[-1] != NUM_MHR_MODEL_PARAMS:
            raise ValueError(
                "MHR pose parameters must have shape (204,) or (B,204), got "
                f"{tuple(model_params.shape)}"
            )
        batch_size = model_params.shape[0]
        model_device = self._model_device()
        model_params = model_params.to(model_device)
        shape = self.shape_params.expand(batch_size, -1).to(model_device)
        expr = self._expr_params.expand(batch_size, -1).to(model_device)
        return mhr_lbs.mhr_query(
            self.model,
            shape,
            model_params,
            expr_params=expr,
            device=str(model_params.device),
        )

    # --- BodyModel interface -------------------------------------------------

    def rest_vertices(self) -> torch.Tensor:
        """Return the shaped canonical mesh in the configured canonical pose."""
        vertices = self._query(self._canonical_model_params)["verts"][0]
        return vertices[self.hand_vertex_mask().to(vertices.device)] if self.hand_only else vertices

    def faces(self) -> npt.NDArray[np.integer[Any]]:
        faces = self._faces.detach().cpu().numpy()
        return self.select_vertex_faces(faces, self.hand_vertex_mask()) if self.hand_only else faces

    def skinning_weights(self) -> torch.Tensor:
        return self._skinning_weights[self.hand_vertex_mask()] if self.hand_only else self._skinning_weights

    def joint_parents(self) -> npt.NDArray[np.integer[Any]]:
        return self._joint_parents.detach().cpu().numpy()

    def face_region_mask(self) -> npt.NDArray[np.bool_]:
        mask = self._face_region_mask
        if self.hand_only:
            mask = mask[self.hand_vertex_mask()]
        return mask.detach().cpu().numpy()

    def hand_vertex_mask(self) -> torch.Tensor:
        # Complete palm/finger subtrees in the public 127-joint MHR rig.
        hand_joints = (*range(42, 65), *range(78, 101))
        return self._skinning_weights[:, hand_joints].sum(dim=1) >= 0.5

    def hand_pose_parameter_mask(self) -> torch.Tensor:
        mask = torch.zeros(NUM_MHR_MODEL_PARAMS, dtype=torch.bool)
        mask[MHR_HAND_DIMS] = True
        return mask

    def hand_joint_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_joints, dtype=torch.bool)
        mask[[*range(41, 65), *range(77, 101)]] = True
        return mask

    def pose_transforms(
        self, pose_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self._query(pose_params)
        batch_size = target["joint_pos"].shape[0]
        canonical_params = self._canonical_model_params.unsqueeze(0).expand(batch_size, -1)
        canonical = self._query(canonical_params)
        transforms = mhr_lbs.joint_relative_transforms(
            target["joint_pos"],
            target["joint_rotmat"],
            canonical["joint_pos"],
            canonical["joint_rotmat"],
        )
        return target["joint_pos"], target["joint_rotmat"], transforms

    def subdivide(
        self, mesh: trimesh.Trimesh, n_iters: int
    ) -> tuple[trimesh.Trimesh, npt.NDArray[np.integer[Any]]]:
        if n_iters < 0:
            raise ValueError(f"n_iters must be non-negative, got {n_iters}")
        original_vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertices = original_vertices
        faces = np.asarray(mesh.faces, dtype=np.int64)
        for _ in range(n_iters):
            vertices, faces = trimesh.remesh.subdivide(vertices, faces)
        parent_idx = cKDTree(original_vertices).query(vertices, k=1)[1].astype(np.int64)
        return (
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            parent_idx,
        )

    def cto_parameters(self) -> dict[str, nn.Parameter]:
        # MHR pose/shape correction remains an application-level concern in
        # the native MHR pipeline. The shape parameter is exposed for callers
        # that intentionally optimize it, but is not silently added to a CTO
        # optimizer by this adapter.
        return {}

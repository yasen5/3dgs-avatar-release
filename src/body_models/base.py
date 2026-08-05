"""Common interface over a parametric human body model + its LBS rig.

Both the existing MHR rig and the new SMPL-X backend implement this, so
model code added for GA-Avatar (triplane binding, GeoNet/RGBNet, the
exact-vertex rigid deformer, the Laplacian/normal losses) is written once
against `BodyModel`, never against SMPL-X- or MHR-specific internals.

The rest of the codebase does not consume `BodyModel` instances directly in
most places -- it reads the `CanonicalMetadata`/`ModelMetadata` dict that a
dataset's `get_metadata()` builds (see src/body_models/metadata.py), which is
in turn populated by calling into a `BodyModel`. Modules that need an actual
mesh operation (subdivide, laplacian, sample) reach it via
`metadata['body_model']`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import trimesh


class BodyModel(ABC):
    @abstractmethod
    def rest_vertices(self) -> torch.Tensor:
        """Canonical (rest-pose) mesh vertices, (V,3), reflecting this body
        model's current learnable shape/offset parameters (if any)."""

    @abstractmethod
    def faces(self) -> npt.NDArray[np.integer[Any]]:
        """(F,3) triangle vertex indices, matching rest_vertices()'s order."""

    @abstractmethod
    def skinning_weights(self) -> torch.Tensor:
        """(V,J) dense per-vertex skinning weights, matching rest_vertices()'s order."""

    @abstractmethod
    def joint_parents(self) -> npt.NDArray[np.integer[Any]]:
        """(J,) kinematic-tree parent index per joint (root may be -1 or 0)."""

    @abstractmethod
    def pose_transforms(
        self, pose_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a pose and return ``(joint_pos, joint_rotmat, transforms)``.

        The first two outputs are global joint quantities. ``transforms`` are
        the per-joint 4x4 transforms from this model's canonical vertex frame
        to the requested pose. The pose parameter layout is backend-specific
        (SMPL-X axis-angle or MHR model parameters), but the outputs are
        deliberately not.
        """

    def lbs(
        self,
        pose_params: torch.Tensor,
        lbs_weights: torch.Tensor | None = None,
        *,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pose canonical vertices with linear blend skinning.

        ``pose_params`` is passed to :meth:`pose_transforms`. By default the
        model's dense weights are used; ``lbs_weights`` can replace them with
        ``(V,J)`` or batched ``(B,V,J)`` weights. ``weights`` is accepted as a
        readable alias for callers that prefer the shorter spelling.
        """
        if lbs_weights is not None and weights is not None:
            raise ValueError("Pass either lbs_weights or weights, not both")
        _, _, bone_transforms = self.pose_transforms(pose_params)
        posed = skin_vertices(
            self,
            bone_transforms,
            lbs_weights if lbs_weights is not None else weights,
        )
        return posed[0] if pose_params.ndim == 1 and posed.ndim == 3 else posed

    def posed_vertices(
        self,
        pose_params: torch.Tensor,
        lbs_weights: torch.Tensor | None = None,
        *,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Explicit alias for :meth:`lbs` for mesh-oriented call sites."""
        return self.lbs(pose_params, lbs_weights, weights=weights)

    def joint_transforms(
        self, pose_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Readable alias for :meth:`pose_transforms`."""
        return self.pose_transforms(pose_params)

    @abstractmethod
    def subdivide(
        self, mesh: trimesh.Trimesh, n_iters: int
    ) -> tuple[trimesh.Trimesh, npt.NDArray[np.integer[Any]]]:
        """Subdivide `mesh` `n_iters` times. Returns the subdivided mesh plus a
        (V_new,) int array giving, for every new vertex, the index (into the
        input mesh's vertex array) of its nearest parent-triangle vertex --
        used to propagate per-base-vertex data (skinning weights, region
        masks) onto the new vertex set without re-deriving it from scratch."""

    @abstractmethod
    def face_region_mask(self) -> npt.NDArray[np.bool_]:
        """(V,) boolean mask, True for vertices in the face/head region,
        matching rest_vertices()'s order."""

    def sample(
        self, mesh: trimesh.Trimesh, n: int
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.integer[Any]]]:
        """Uniformly sample `n` points on `mesh`'s surface -> (points (n,3),
        face_idx (n,)). Mirrors the `cano_mesh.sample(...)` call already used
        by src/models/deformer/rigid.py's SkinningField.sample_skinning_loss.
        Default implementation delegates to trimesh; override only if a
        backend needs something else."""
        points, face_idx = mesh.sample(n, return_index=True)
        return np.asarray(points, dtype=np.float32), np.asarray(face_idx)

    def laplacian(self, mesh: trimesh.Trimesh) -> Any:
        """Uniform-weight mesh-graph Laplacian operator (scipy.sparse, V x V)
        over `mesh`'s vertices, for GA-Avatar's L_lap smoothness term. Default:
        trimesh's own uniform Laplacian; override only if a backend needs
        cotangent weights."""
        return trimesh.smoothing.laplacian_calculation(mesh)

    def cto_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Learnable Canonical-Template-Optimization parameters (shape, joint
        offset, face offset, ...) owned by this body model, keyed by name.
        Backends without a CTO stage (e.g. MHR) return {}."""
        return {}


def skin_vertices(
    body_model: BodyModel,
    bone_transforms: torch.Tensor,
    lbs_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply per-joint transforms to the model's canonical vertices.

    ``bone_transforms`` may be ``(J, 4, 4)`` or ``(B, J, 4, 4)`` and follows
    the column-vector convention used by :meth:`BodyModel.lbs`. ``lbs_weights``
    may be omitted, supplied as ``(V,J)``, or supplied per batch as ``(B,V,J)``.
    The returned shape is ``(V,3)`` for a single transform and ``(B,V,3)`` for
    batched transforms.
    """
    single_transform = bone_transforms.ndim == 3
    if bone_transforms.ndim == 3:
        bone_transforms = bone_transforms.unsqueeze(0)
    if bone_transforms.ndim != 4:
        raise ValueError(
            "bone_transforms must have shape (J,4,4) or (B,J,4,4), got "
            f"{tuple(bone_transforms.shape)}"
        )

    vertices = body_model.rest_vertices()
    weights = body_model.skinning_weights() if lbs_weights is None else lbs_weights
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"rest_vertices() must return (V,3), got {tuple(vertices.shape)}")
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    elif weights.ndim != 3:
        raise ValueError(
            "lbs_weights must have shape (V,J) or (B,V,J), got "
            f"{tuple(weights.shape)}"
        )
    if weights.shape[0] == 1 and bone_transforms.shape[0] != 1:
        weights = weights.expand(bone_transforms.shape[0], -1, -1)
    expected = (bone_transforms.shape[0], vertices.shape[0], bone_transforms.shape[1])
    if weights.shape != expected:
        raise ValueError(
            "lbs_weights and body model disagree: expected "
            f"(V,J)={(vertices.shape[0], bone_transforms.shape[1])} or "
            f"(B,V,J)={expected}, got {tuple(weights.shape)}"
        )

    weights = weights.to(device=bone_transforms.device, dtype=bone_transforms.dtype)
    vertices = vertices.to(device=bone_transforms.device, dtype=bone_transforms.dtype)

    homogeneous = torch.cat(
        [vertices, torch.ones_like(vertices[:, :1])], dim=1
    )
    transformed = torch.einsum(
        "bjkl,vl->bjvk", bone_transforms, homogeneous
    )[..., :3]
    posed = torch.einsum("bvj,bjvc->bvc", weights, transformed)
    return posed[0] if single_transform else posed

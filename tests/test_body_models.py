from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import trimesh

from src.body_models.base import BodyModel


class _ToyBodyModel(BodyModel):
    def __init__(self) -> None:
        self._vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self._weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def rest_vertices(self) -> torch.Tensor:
        return self._vertices

    def faces(self) -> npt.NDArray[np.integer[Any]]:
        return np.array([[0, 1, 1]], dtype=np.int64)

    def skinning_weights(self) -> torch.Tensor:
        return self._weights

    def joint_parents(self) -> npt.NDArray[np.integer[Any]]:
        return np.array([0, 0], dtype=np.int64)

    def pose_transforms(
        self, pose_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pose_params.ndim == 1:
            pose_params = pose_params.unsqueeze(0)
        batch_size = pose_params.shape[0]
        transforms = torch.eye(4).repeat(batch_size, 2, 1, 1)
        transforms[:, 1, 0, 3] = 2.0
        joint_pos = transforms[:, :, :3, 3]
        joint_rotmat = transforms[:, :, :3, :3]
        return joint_pos, joint_rotmat, transforms

    def subdivide(
        self, mesh: trimesh.Trimesh, n_iters: int
    ) -> tuple[trimesh.Trimesh, npt.NDArray[np.integer[Any]]]:
        return mesh, np.arange(len(mesh.vertices), dtype=np.int64)

    def face_region_mask(self) -> npt.NDArray[np.bool_]:
        return np.zeros(2, dtype=bool)


def test_lbs_uses_default_weights_and_returns_unbatched_vertices() -> None:
    model = _ToyBodyModel()

    posed = model.lbs(torch.zeros(1))

    assert posed.shape == (2, 3)
    assert torch.allclose(posed, torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))


def test_lbs_accepts_custom_and_batched_weights() -> None:
    model = _ToyBodyModel()
    swapped = model.skinning_weights().flip(1)

    posed = model.lbs(torch.zeros(2, 1), lbs_weights=swapped)

    expected_one = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert posed.shape == (2, 2, 3)
    assert torch.allclose(posed[0], expected_one)
    assert torch.allclose(posed[1], expected_one)


class _ToyResidualBodyModel(_ToyBodyModel):
    def pose_residuals(self, pose_params: torch.Tensor) -> torch.Tensor:
        if pose_params.ndim == 1:
            pose_params = pose_params.unsqueeze(0)
        residual = torch.tensor(
            [[0.0, 0.5, 0.0], [-0.25, 0.0, 0.0]],
            dtype=self._vertices.dtype,
            device=self._vertices.device,
        )
        return residual.unsqueeze(0).expand(pose_params.shape[0], -1, -1)


def test_lbs_adds_backend_pose_residuals() -> None:
    model = _ToyResidualBodyModel()

    posed = model.lbs(torch.zeros(1))

    expected = torch.tensor([[0.0, 0.5, 0.0], [2.75, 0.0, 0.0]])
    assert posed.shape == (2, 3)
    assert torch.allclose(posed, expected)


class _ToyHandBodyModel(_ToyBodyModel):
    def hand_vertex_mask(self) -> torch.Tensor:
        return torch.tensor([False, True])

    def hand_pose_parameter_mask(self) -> torch.Tensor:
        return torch.tensor([False, True])

    def hand_joint_mask(self) -> torch.Tensor:
        return torch.tensor([False, True])


def test_hand_side_defaults_to_none_and_rejects_invalid_values() -> None:
    model = _ToyHandBodyModel()
    assert model.hand_side is None

    model.set_hand_only(True, side="left")
    assert model.hand_side == "left"

    model.set_hand_only(False)
    assert model.hand_side is None

    try:
        model.set_hand_only(True, side="both")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid side")

    try:
        model.set_hand_only(False, side="left")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when side is set without enabled")


def test_hand_interface_exposes_exact_vertices_and_freezes_other_pose_gradients() -> None:
    model = _ToyHandBodyModel()
    pose = torch.tensor([[2.0, 3.0]], requires_grad=True)

    assert torch.equal(model.hand_vertices(), torch.tensor([[1.0, 0.0, 0.0]]))
    restricted = model.freeze_non_hand_pose(pose)
    restricted.sum().backward()

    assert pose.grad is not None
    assert torch.equal(pose.grad, torch.tensor([[0.0, 1.0]]))


class _ToyBodyPartBodyModel(_ToyBodyModel):
    def body_part_vertex_mask(self, part: str) -> torch.Tensor:
        if part == "left_hand":
            return torch.tensor([False, True])
        raise ValueError(part)


def test_body_part_vertex_mask_defaults_to_not_implemented() -> None:
    model = _ToyBodyModel()
    try:
        model.body_part_vertex_mask("head")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for a backend without body parts")


def test_body_part_vertices_selects_from_full_topology_regardless_of_hand_only() -> None:
    model = _ToyBodyPartBodyModel()
    model._hand_only = True  # simulate hand-only mode without a full hand implementation

    assert torch.equal(model.body_part_vertices("left_hand"), torch.tensor([[1.0, 0.0, 0.0]]))
    # hand_only must still be in effect afterwards.
    assert model.hand_only


def test_hand_face_selection_remaps_to_compact_topology() -> None:
    faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]], dtype=np.int64)
    selected = BodyModel.select_vertex_faces(
        faces, torch.tensor([False, True, True, True])
    )

    assert np.array_equal(selected, np.array([[0, 1, 2]], dtype=np.int64))

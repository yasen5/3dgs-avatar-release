import numpy as np

from scripts.select_unoccluded_hands import (
    ray_hits_before_vertex,
    rays_hit_before_vertices,
)
from src.body_models.mhr_body_model import _descendants


def test_descendants_returns_only_the_requested_joint_subtree():
    parents = np.array([-1, 0, 0, 1, 1, 2, 3], dtype=np.int64)
    assert _descendants(parents, 1) == (1, 3, 4, 6)


def test_ray_hit_must_be_strictly_before_the_hand_vertex():
    hand_vertex = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    blocker = np.array(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    face = np.array([[0, 1, 2]], dtype=np.int64)
    assert ray_hits_before_vertex(hand_vertex, blocker, face, tolerance=0.002)

    behind = blocker.copy()
    behind[:, 2] = 3.0
    assert not ray_hits_before_vertex(hand_vertex, behind, face, tolerance=0.002)

    at_endpoint = blocker.copy()
    at_endpoint[:, 2] = 2.0
    assert not ray_hits_before_vertex(hand_vertex, at_endpoint, face, tolerance=0.002)


def test_batched_ray_test_checks_all_faces_without_raster_candidates():
    import torch

    hand_vertices = np.array(
        [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float32
    )
    blocker = np.array(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    hits = rays_hit_before_vertices(
        hand_vertices, blocker, faces, tolerance=0.002, device=torch.device("cpu")
    )

    np.testing.assert_array_equal(hits, np.array([True, False]))

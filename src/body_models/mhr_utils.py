"""Small MHR-adapter helpers specific to this repo's pose-encoding/canonicalization
conventions, kept separate from the vendored (symlinked) mhr_lbs.py so that repo stays
the single source of truth for the validated MHR rig math (posed mesh, joint transforms,
dense skinning weights) while this file only adds the glue this codebase's deformer/
pose-correction modules expect.

Split into its own module (rather than living in dataset/mhr_native.py) so that
models/pose_correction/pose_correction.py can import it without a circular dependency:
dataset/__init__.py is pulled in transitively by scene -> models -> pose_correction.
"""
from __future__ import annotations

import torch

from src.constants import (
    BIG_POSE_ARM_DIMS,
    BIG_POSE_DELTA,
    BIG_POSE_LEG_DIMS,
    BIG_POSE_ZERO_PREFIX_LEN,
    BODY_POSE_OFFSET,
)

# Empirically-found "big pose" (SMPL star/Vitruvian-pose analogue) dims in MHR's
# 204-dim model_params vector: global_trans(3) + global_rot(3) + body_pose(198),
# so +6 offsets into body_pose, delta 0.6 -- found by finite-differencing which
# body_pose dims move which joint chains: not derivable from the rig itself,
# must be reused as-is.
#
# Identified by joint hierarchy, not by these dims' old (wrong) names: dims
# (24,34) move the chain rooted at joints 38/74, whose root sits at shoulder
# height (right beside the head joint) and terminates in a 5-digit hand --
# these are the ARM dims. Dims (45,54) move the chain rooted at joints 18/2,
# whose root sits exactly at pelvis height -- these are the LEG dims. (A prior
# version of this file had the two pairs swapped/named backwards, which is how
# a real bug in the leg dims went unnoticed while the arm dims got "fixed"
# instead.)
#
# Both pairs need -BIG_POSE_DELTA, not +, to abduct outward without crossing
# the midline. +BIG_POSE_DELTA on the leg dims is severe enough that each
# foot joint ends up on the OPPOSITE side of the body from its own hip root
# (e.g. left hip root at x=-0.078 swings to a foot at x=+0.287) -- a genuine
# kinematic crossing, not just a tight stance -- which is what actually
# produced the cross-legged canonical pose. -BIG_POSE_DELTA keeps each leg's
# whole chain on its own side (verified with fcl: zero triangle-triangle
# collision between the left/right leg surfaces, 0/3579 vertices cross
# sides, ~7.7cm minimum surface gap) while abducting into a clean
# star/Vitruvian pose.
# (BIG_POSE_ARM_DIMS, BIG_POSE_LEG_DIMS, BIG_POSE_DELTA live in src/constants.py)


def build_big_pose_model_params(reference_model_params: torch.Tensor) -> torch.Tensor:
    mp = reference_model_params.clone()
    mp[:BIG_POSE_ZERO_PREFIX_LEN] = 0.0
    for dim in BIG_POSE_ARM_DIMS + BIG_POSE_LEG_DIMS:
        mp[BODY_POSE_OFFSET + dim] = -BIG_POSE_DELTA
    return mp


def local_joint_rotmats(joint_rotmat: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """(B,J,3,3) global per-joint rotmats -> (B,J,3,3) LOCAL (parent-relative),
    with joint 0 (root) forced to identity -- matches this codebase's `pose_rot`
    convention: the pose encoder should see body articulation only, not the
    person's facing direction."""
    j = joint_rotmat.shape[1]
    local = torch.empty_like(joint_rotmat)
    local[:, 0] = torch.eye(3, device=joint_rotmat.device, dtype=joint_rotmat.dtype)
    for idx in range(1, j):
        p = int(parents[idx].item())
        local[:, idx] = torch.matmul(joint_rotmat[:, p].transpose(-1, -2), joint_rotmat[:, idx])
    return local

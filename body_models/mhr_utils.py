"""Small MHR-adapter helpers specific to this repo's pose-encoding/canonicalization
conventions, kept separate from the vendored (symlinked) mhr_lbs.py so that repo stays
the single source of truth for the validated MHR rig math (posed mesh, joint transforms,
dense skinning weights) while this file only adds the glue this codebase's deformer/
pose-correction modules expect.

Split into its own module (rather than living in dataset/mhr_native.py) so that
models/pose_correction/pose_correction.py can import it without a circular dependency:
dataset/__init__.py is pulled in transitively by scene -> models -> pose_correction.
"""
import torch

# Empirically-found "big pose" (SMPL star/Vitruvian-pose analogue) dims in MHR's
# 204-dim model_params vector: global_trans(3) + global_rot(3) + body_pose(198),
# so +6 offsets into body_pose. shoulder pair (45,54), hip pair (24,34), delta 0.6 --
# found by finite-differencing which body_pose dims abduct the shoulders/hips: not
# derivable from the rig itself, must be reused as-is.
BIG_POSE_SHOULDER_DIMS = [45, 54]
BIG_POSE_HIP_DIMS = [24, 34]
BIG_POSE_DELTA = 0.6


def build_big_pose_model_params(reference_model_params: torch.Tensor) -> torch.Tensor:
    mp = reference_model_params.clone()
    mp[:136] = 0.0
    for dim in BIG_POSE_SHOULDER_DIMS + BIG_POSE_HIP_DIMS:
        mp[6 + dim] = BIG_POSE_DELTA
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

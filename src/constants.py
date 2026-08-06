"""Fixed, non-tunable numeric/structural constants used across src/.

Anything that represents a real experimental choice belongs in a yaml config
under configs/, not here -- this module is only for values that are either
mathematically fixed (e.g. a quaternion identity component) or physically
fixed to this specific rig/dataset (e.g. MHR's parameter-vector layout) and
that no researcher would plausibly want to override per-run.

Note: src/body_models/mhr_lbs.py is a symlinked adapter shared byte-for-byte
with the sibling gauhuman_baseline project and is deliberately NOT
consolidated here -- its own module-level constants (NUM_VERTS, NUM_JOINTS,
_FLIP, unit-scale factor, etc.) must stay put to preserve that shared
contract.
"""
from __future__ import annotations

import math

# --- Numerical-stability epsilons -------------------------------------------
OPACITY_LOSS_EPS = 1e-6
MIN_DIST2 = 0.0000001
ADAM_EPS = 1e-15
SCALE_EXP_MIN_EPS = 1e-6
DIR_NORM_EPS = 1e-12
MASK_LOSS_BCE_EPS = 1e-3

# --- Fixed geometry / rig dimensions -----------------------------------------
GAUSSIAN_XYZ_DIM = 3
GAUSSIAN_SCALE_DIM = 3
GAUSSIAN_ROT_DIM = 4  # quaternion (w,x,y,z)
ROTMAT_FLAT_DIM = 9  # flattened 3x3 rotation matrix
JOINT_POS_DIM = 3
JOINT_FEAT_BASE_DIM = ROTMAT_FLAT_DIM + JOINT_POS_DIM + 1  # rot(9) + pos(3) + bone_len(1)
RGB_DIM = 3
COV_UPPER_TRI_DIM = 6  # upper triangle of a symmetric 3x3 covariance matrix

MHR_SHAPE_DIM = 45
MHR_MODEL_PARAMS_DIM = 204
BODY_POSE_OFFSET = 6  # global_trans(3) + global_rot(3) prefix into model_params' body_pose block
BIG_POSE_ZERO_PREFIX_LEN = 136  # global_trans(3) + global_rot(3) + leading body_pose dims zeroed in the big pose

# --- Empirically-derived MHR "big pose" (Vitruvian/star pose) constants -----
# See src/body_models/mhr_utils.py::build_big_pose_model_params for the full
# derivation -- these dims/delta were found by finite-differencing which
# body_pose dims move which joint chains and are not derivable from the rig
# itself, so they must be reused as-is.
BIG_POSE_ARM_DIMS = [24, 34]
BIG_POSE_LEG_DIMS = [45, 54]
BIG_POSE_DELTA = 0.6

# --- MHR hand DOFs (see scripts/refine_hands_rtmpose.py for the derivation) -------------------
# `mhr_param_hand_idxs` from third_party/mesh_recovery/sam3d-body-original's own
# sam_3d_body/models/modules/mhr_utils.py -- offsets (+BODY_POSE_OFFSET, same convention as
# BIG_POSE_ARM_DIMS/BIG_POSE_LEG_DIMS above) into model_params' body_pose block for the 54 hand
# DOFs (27 per hand: 1 wrist + 5 fingers x 4, in per-joint XYZ-euler-ish order), verified by
# diffing a real fit's mhr_model_params[BODY_POSE_OFFSET:BODY_POSE_OFFSET+133] against its own
# body_pose_params array: they agree everywhere except exactly these 54 indices.
MHR_HAND_DIMS = [
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
]

# Per-hand split of MHR_HAND_DIMS, for single-hand training (see
# BodyModel.set_hand_only's `side` argument). Found by finite-differencing
# each dim of a real fit's mhr_model_params (perturb by +0.3, re-run
# mhr_query) and checking which hand's hand_vertex_mask (mhr_body_model.py)
# moved: dims 68-94 move only the joints-42-64 (left) hand, dims 95-115 move
# only the joints-78-100 (right) hand. Dims 62-67 move neither hand under
# this test (apparently unused DOFs) and are intentionally in neither list --
# both-hands mode (MHR_HAND_DIMS above) still includes them for parity with
# its pre-existing behavior.
MHR_HAND_DIMS_LEFT = tuple(range(68, 95))
MHR_HAND_DIMS_RIGHT = tuple(range(95, 116))

# MHR joint indices (into mhr_lbs.mhr_query's 127-joint output) for each hand's 21-keypoint
# wrist->finger chain, in standard wrist-first order (0=wrist,1-4=thumb,...,17-20=pinky) --
# found by walking joint_parents from the wrist down each of the 5 finger chains and confirmed
# to exact sub-mm/sub-pixel agreement against a saved fit's own pred_keypoints_3d/2d. This is
# also rtmlib's native (to_openpose=False) hand21 keypoint order, so RTMPose output maps to
# these joints index-for-index with no relabeling.
MHR_HAND_A_JOINTS = [41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64]
MHR_HAND_B_JOINTS = [j + 36 for j in MHR_HAND_A_JOINTS]

# --- Dataset / scene-scale constants -----------------------------------------
# Scene-scale constant for this data's camera rig (verified to work well at
# this data scale) -- not a per-camera-rig quantity, so it's fixed here rather
# than per-dataset-config.
CAMERAS_EXTENT = 3.469298553466797

# --- Fixed math/algorithmic conventions --------------------------------------
SH_DC_OFFSET = 0.5  # standard SH-to-RGB DC offset convention
QUATERNION_IDENTITY_W = 1.0  # [1,0,0,0] identity rotation quaternion's w component
LAST_LAYER_INIT_STD = 1e-5  # near-zero init std for a network's optional last-layer init
SKIP_CONNECTION_SCALE = 1.0 / math.sqrt(2)  # standard SIREN/IDR-style skip-connection rescale

# --- Cosmetic/logging-only constants (not real experiment choices) ----------
LOSS_EMA_ALPHA = 0.4
PROGRESS_BAR_UPDATE_INTERVAL = 10
PROGRESS_BAR_PRECISION = 7
TRAIN_VAL_SUBSAMPLE_STRIDE = 10

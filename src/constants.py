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

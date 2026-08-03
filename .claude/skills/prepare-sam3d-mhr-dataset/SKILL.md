---
name: prepare-sam3d-mhr-dataset
description: Prepare a fresh monocular video or frame sequence for the MHR-native 3DGS-Avatar pipeline by running SAM3D-Body, feeding raw results to an external MHR runner when needed, and producing the train/test images, masks, and mhr_parms.pth layout consumed by MHRNativeDataset. Use when starting a new subject or deciding between dense and keyframe MHR inference.
---

# Prepare a SAM3D-Body / MHR dataset

This repository consumes native MHR parameters. Do not insert a SMPL fitting or
SMPL-to-MHR conversion step unless the task explicitly requires one.

The preparation scripts live in the workspace pipeline tree, not in this Git
checkout:

```text
pipelines/texture_appearance/threedgs_avatar_mhr/
  prepare_mhr_dense_track.py
  prepare_mhr_keyframe_track.py
```

Set `PIPELINE_ROOT` to the workspace root containing `pipelines/` if it is not
already the current workspace root. Before running, inspect the script's
imports and use the environment that contains `uvpipeline`, the original
SAM3D-Body predictor, YOLO's person detector, and the required CUDA packages.

Required model assets include the TorchScript MHR model, normally:

```text
<sam3d-checkpoints>/sam-3d-body-dinov3/assets/mhr_model.pt
```

## Choose the track

- Use **dense** inference for a new dataset when GPU time is available. SAM3D-
  Body runs on every working frame, so pose discontinuities are image-driven
  rather than interpolation artifacts. The script generates Sapiens2 person
  masks and stores them as `valid_masks/`.
- Use **keyframe** inference when SAM3D-Body cost must be reduced. It runs on
  sparse keyframes, interpolates `model_params` (branch-preserving SLERP for
  global rotation and linear interpolation for the remaining MHR pose vector),
  and initializes non-keyframe shape from the nearest keyframe. Its
  `valid_masks/` are the skin plus projected head/neck supervision region.

Do not mix the two mask semantics without recording it in the manifest.

## Dense preparation

From the workspace root, use the actual pipeline script. If the current
directory is this nested checkout, replace the script path with an absolute
path to the workspace's `pipelines/` directory:

```bash
python /absolute/path/to/workspace/pipelines/texture_appearance/threedgs_avatar_mhr/prepare_mhr_dense_track.py \
  --video /absolute/path/to/subject.mp4 \
  --out-dir /absolute/path/to/mhr_native_subject_dense \
  --frame-stride 1 \
  --test-every 5 \
  --scale 0.2 \
  --device cuda
```

`--frame-stride` may be increased to reduce the working set. `--test-every 5`
places every fifth working frame in test and the rest in train. Keep the same
image scale for images, masks, focal length, width, and height; the script
does this together. The current dense script does not expose `--mhr-model`;
its SAM3D-Body/MHR model location comes from the external runner environment,
so inspect that runner's configuration before starting inference.

## Keyframe preparation

```bash
python pipelines/texture_appearance/threedgs_avatar_mhr/prepare_mhr_keyframe_track.py \
  --video /absolute/path/to/subject.mp4 \
  --out-dir /absolute/path/to/mhr_native_subject_keyframes \
  --mhr-model /absolute/path/to/assets/mhr_model.pt \
  --frame-stride 3 \
  --keyframe-stride 15 \
  --test-every 5 \
  --scale 0.2 \
  --device cuda
```

Keyframe spacing is in original-video frame indices, while `frame-stride`
controls the working dataset. Ensure the first and last usable frames are
covered. Inspect keyframe overlays before launching a long training run.

## Contract consumed by this checkout

The output root must have this prepared layout:

```text
<root>/
  train/
    images/<frame-id>.png
    valid_masks/<frame-id>.png
    mhr_parms.pth
  test/
    images/<frame-id>.png
    valid_masks/<frame-id>.png
    mhr_parms.pth
  manifest.json
```

`mhr_parms.pth` must contain:

```text
shape_params:   (N, 45) float32
model_params:   (N, 204) float32
cam_t:          (N, 3) float32
focal_length:   (N,) float32
width, height:  (N,) integer
frame_ids:      N ids matching image and mask filenames
```

Keyframe outputs also carry `is_keyframe`. Every frame must have finite
`shape_params`, `model_params`, and `cam_t`; every image and mask must exist
with matching dimensions. `valid_masks/<id>.png` is binary: nonzero pixels are
the region used by `MHRNativeDataset` both for compositing and alpha loss.

The raw SAM3D-Body fit files should remain available under the preparation
output (`raw_sam3d/` or `raw_sam3d_keyframes/`). A runner sometimes called
`mhr_runner` must preserve, at minimum, `shape_params`, `mhr_model_params`,
`pred_cam_t`/`cam_t`, `focal_length`, `width`, and `height`. This checkout has
no tracked executable named `mhr_runner`; locate the external runner with
`rg --files` and inspect its CLI rather than inventing a command. The native
adapter is `src/body_models/mhr_utils.py` plus the external MHR model.

## Point the avatar config at the result

Use a dataset config with:

```yaml
dataset:
  name: mhr_native
  source_layout: prepared
  root_dir: /absolute/path/to/mhr_native_subject_dense
  mhr_model_path: /absolute/path/to/assets/mhr_model.pt
  mhr_device: cuda
```

Copy an existing file under `configs/dataset/` and change only the dataset
name, root, and model path. Then pass it as `dataset=<new-config-name>`.

## Validate before training

Run these read-only checks, adapting the root path:

```bash
find /absolute/path/to/mhr_native_subject_dense/train -type f | head
find /absolute/path/to/mhr_native_subject_dense/test -type f | head
python - <<'PY'
from pathlib import Path
import torch

root = Path('/absolute/path/to/mhr_native_subject_dense')
for split in ('train', 'test'):
    p = torch.load(root / split / 'mhr_parms.pth', weights_only=False)
    print(split, {k: tuple(v.shape) if hasattr(v, 'shape') else len(v)
                  for k, v in p.items()})
    assert len(p['frame_ids']) == p['shape_params'].shape[0]
    assert p['shape_params'].shape[1] == 45
    assert p['model_params'].shape[1] == 204
    assert p['cam_t'].shape[1] == 3
PY
```

Also inspect several source-image/mask pairs. A zero-area mask, mismatched
frame IDs, a missing raw fit, or a visibly wrong camera translation should be
fixed before `scripts/train.py` is started.

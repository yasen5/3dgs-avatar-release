---
name: debug-mhr-dataset
description: Diagnose MHR dataset preparation, masks, camera projection, and SAM3D-Body fits for this 3DGS-Avatar checkout. Use when training fails at data loading or rendered meshes/images look misaligned.
---

# Debug MHR data

Debug the data contract before changing the avatar model. The most common
failures are missing per-frame fields, image/mask dimension mismatches, a
wrong focal length or `cam_t`, and masks that supervise the wrong pixels.

## Validate a prepared root

For `source_layout: prepared`, check both splits:

```bash
python - <<'PY'
from pathlib import Path
import cv2
import torch

root = Path('/absolute/path/to/mhr_native_subject')
for split in ('train', 'test'):
    d = root / split
    p = torch.load(d / 'mhr_parms.pth', weights_only=False)
    ids = list(p['frame_ids'])
    assert p['shape_params'].shape == (len(ids), 45)
    assert p['model_params'].shape == (len(ids), 204)
    assert p['cam_t'].shape == (len(ids), 3)
    for fid in ids[:5]:
        image = cv2.imread(str(d / 'images' / f'{fid}.png'))
        mask = cv2.imread(str(d / 'valid_masks' / f'{fid}.png'), cv2.IMREAD_GRAYSCALE)
        assert image is not None and mask is not None, fid
        assert image.shape[:2] == mask.shape[:2], fid
    print(split, len(ids), 'frames; sample image/mask sizes agree')
PY
```

Check that the parameters are finite and that the same IDs occur in the image,
mask, and parameter files. For keyframe data, inspect `is_keyframe` and the
manifest's interpolation settings.

## Inspect masks

For a ZJU camera layout, regenerate Sapiens2 person masks with the repository
script when the existing masks are suspect:

```bash
python scripts/gen_sapiens2_masks.py \
  --images /absolute/path/to/CoreView_377/Camera_B1/images \
  --output /absolute/path/to/CoreView_377/Camera_B1/masks \
  --overlay-dir /absolute/path/to/mask-overlays
```

Do not silently substitute `masks/` for `valid_masks/` in a prepared dataset:
the prepared reader expects `valid_masks/`, while the ZJU reader uses the
configured `mask_dir` (normally `masks`).

## Smoke-test SAM3D-Body on ZJU

The repository includes a non-destructive five-frame smoke test. It writes to
a separate output directory and does not modify the training data:

```bash
python scripts/smoke_sam3d_body_zju377_no_hands.py \
  --data-root /absolute/path/to/CoreView_377 \
  --camera Camera_B1 \
  --num-frames 5 \
  --output-dir /absolute/path/to/sam3d-smoke \
  --device cuda
```

Use this to distinguish predictor/environment failures from dataset-layout
failures. For a fresh video, use the external dense or keyframe preparation
skill instead; this smoke test is ZJU-specific.

## Projection overlays

For a ZJU subject with native SMPL comparison data, use:

```bash
python scripts/render_zju_mhr_smplx_overlay.py \
  --subject-dir /absolute/path/to/CoreView_377 \
  --output-dir /absolute/path/to/mhr-vs-smpl \
  --start-frame 0 --end-frame 570 --stride 10
```

The MHR projection uses the raw fit's `vertices + pred_cam_t`; it does not use
ZJU's multi-camera extrinsics for the MHR side. If the MHR overlay is shifted,
check `pred_cam_t`, `focal_length`, image scale, and the selected camera before
checking the renderer.

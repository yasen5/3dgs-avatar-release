---
name: export-checkpoint-ply
description: Export a trained 3DGS-Avatar MHR checkpoint (.pth) to a standard 3DGS PLY, or verify/reload that PLY with the matching converter weights. Use when a checkpoint must be viewed in SuperSplat, MeshLab, SIBR, or another generic Gaussian viewer.
---

# Export a checkpoint to PLY

Run commands from the `3dgs-avatar-release` repository root. This repository's
checkpoint is a five-item tuple saved by `src/scene/__init__.py`; it is not the
same checkpoint format as the neighboring Gaussian-avatar projects.

## Before exporting

Identify all of the following:

- the checkpoint, normally `exp/<run>/ckpt<iteration>.pth`;
- the matching dataset config, including its MHR model path and dataset root;
- the iteration encoded by the checkpoint;
- the exact model/deformer/texture/pose-correction Hydra overrides used to train.

The checkpoint does not contain a complete Hydra config. Prefer the run's
`exp/<run>/.hydra/config.yaml` and `.hydra/overrides.yaml`, or reproduce their
overrides explicitly. A mismatched architecture can fail at `load_state_dict`
or, worse, produce an invalid-looking export.

## Export

For a run whose config is already represented by a dataset config, use:

```bash
python scripts/export_ply.py \
  dataset=trimmed_final_mhr \
  exp_dir=/absolute/path/to/exp/my-run \
  load_ckpt=/absolute/path/to/exp/my-run/ckpt15000.pth \
  opt.iterations=15000 \
  wandb_disable=true
```

If the run used custom model settings, add the same overrides from the saved
Hydra files, for example `rigid=... non_rigid=... texture=...` and the relevant
`pose_correction=...` override. `opt.iterations` controls both the checkpoint's
default lookup and the output directory name; it must match the checkpoint
iteration for a normal export.

The output is:

```text
<exp_dir>/point_cloud/iteration_<iteration>/point_cloud.ply
```

`export_ply.py` loads the checkpoint, evaluates the neural texture for one
reference test view, and writes that view's representative RGB as well as the
latent Gaussian fields. If `color_ref_view` is set, it must match an image name
in the selected test dataset; otherwise the first test view is used.

## What the PLY means

The RGB in this PLY is a baked snapshot, not the model's complete appearance:
the trained texture network remains view- and pose-dependent. Keep the `.pth`
checkpoint and converter weights when exact reconstruction is needed.

Do not use an exporter from another repository unless its checkpoint schema is
confirmed. In particular, do not pass this `.pth` to an exporter expecting an
`avatar_state_dict` or an MHR pose `.npz`.

## Verify a PLY plus converter weights

To exercise the reload path that preserves the neural converter, use:

```bash
python scripts/render_from_ply.py \
  dataset=trimmed_final_mhr \
  exp_dir=/absolute/path/to/exp/my-run \
  ply_path=/absolute/path/to/point_cloud.ply \
  load_ckpt=/absolute/path/to/exp/my-run/ckpt15000.pth \
  opt.iterations=15000 \
  suffix=test-from-ply \
  wandb_disable=true
```

The script writes rendered test frames below
`<exp_dir>/test-from-ply/renders`. Compare them with `scripts/render.py` from
the full checkpoint if export fidelity matters.

## Checks

- Confirm the PLY exists and has a nonzero vertex count.
- Confirm the PLY's `iteration_<N>` agrees with the intended checkpoint.
- If loading fails, compare `.hydra/config.yaml` and `.hydra/overrides.yaml`
  before changing code.
- If a generic viewer shows odd colors, remember that the static RGB is only
  the selected reference view; the latent fields are still present for this
  repository's reload path.

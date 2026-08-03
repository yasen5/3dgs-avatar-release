---
name: train-render-mhr-avatar
description: Train, resume or evaluate the MHR-native 3DGS-Avatar model in this repository, including Hydra overrides, checkpoint selection, test rendering, and out-of-distribution pose rendering. Use after a prepared MHR dataset exists.
---

# Train and render an MHR avatar

Run from the repository root. First create or select a dataset config under
`configs/dataset/` whose `root_dir` points to the prepared MHR dataset and whose
`mhr_model_path` points to the same `mhr_model.pt` used during preparation.

Keep the model/deformer/texture/pose-correction overrides identical across
training, export, and rendering. The checkpoint stores weights but not a full
Hydra configuration.

## Smoke test

Before a long run, test dataset loading, MHR forward, canonical point-cloud
creation, and one optimizer step:

```bash
python scripts/train.py \
  dataset=trimmed_final_mhr_dense \
  exp_dir=/absolute/path/to/exp/subject_smoke \
  opt.iterations=1000 \
  save_iterations=[1000] \
  checkpoint_iterations=[1000] \
  test_iterations=[] \
  wandb_disable=true
```

Use the actual dataset config name, and lower `opt.iterations` only for the
smoke test. Confirm that `ckpt1000.pth` and
`point_cloud/iteration_1000/point_cloud.ply` are created before scaling up.

## Train from scratch

```bash
python scripts/train.py \
  dataset=trimmed_final_mhr_dense \
  exp_dir=/absolute/path/to/exp/subject_iter30k \
  opt.iterations=30000 \
  save_iterations=[30000] \
  checkpoint_iterations=[30000] \
  wandb_disable=true
```

The default config trains with the MHR-native deformer and neural texture
settings in `configs/config.yaml`. Override only the experiment choices needed
for the subject, such as `pose_correction=direct`, `texture=...`, or an option
file. Record the final command and Hydra output directory with the run.

## Initialize from an existing checkpoint

Use `start_checkpoint=/path/to/ckpt<N>.pth` when the intent is to initialize a
new run with an existing model. Confirm architecture compatibility first. Do
not assume this is a seamless optimizer resume: inspect `scripts/train.py` and
the checkpoint iteration semantics before claiming a resume, and write to a new
`exp_dir` so the source run is preserved.

## Test rendering

Render a checkpoint on the configured held-out frames:

```bash
python scripts/render.py \
  mode=test \
  dataset=trimmed_final_mhr_dense \
  exp_dir=/absolute/path/to/exp/subject_iter30k \
  load_ckpt=/absolute/path/to/exp/subject_iter30k/ckpt30000.pth \
  opt.iterations=30000 \
  wandb_disable=true
```

Outputs are written under `exp_dir/test-<dataset.test_mode>/renders`, with
metrics in `results.npz` when `evaluate=true`. The `test_every` setting and
dataset split determine which frames are rendered.

For ZJU out-of-distribution sequences, use `mode=predict` only with a dataset
config that supplies the supported prediction sequence and files. This mode is
not a generic fresh-video renderer.

## Checkpoint and config troubleshooting

- `FileNotFoundError` for `mhr_parms.pth`, images, or masks means the dataset
  root/layout is wrong; fix preparation/configuration first.
- `load_state_dict` size errors usually mean the architecture overrides do not
  match the checkpoint. Compare `exp/<run>/.hydra/config.yaml` and
  `.hydra/overrides.yaml`.
- A blank or unstable result can come from incorrect `cam_t`, focal length,
  image scale, or masks. Verify the MHR fit overlays and the prepared
  `mhr_parms.pth` contract before tuning loss weights.
- Keep `wandb_disable=true` for local smoke tests; remove it only when online
  logging is intentionally configured.

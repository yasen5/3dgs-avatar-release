#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import os
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from random import randint
from typing import TypedDict, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    LOSS_EMA_ALPHA,
    MASK_LOSS_BCE_EPS,
    PROGRESS_BAR_PRECISION,
    PROGRESS_BAR_UPDATE_INTERVAL,
    TRAIN_VAL_SUBSAMPLE_STRIDE,
)
from src.dataset.training_interface import crop_hand_regions
from src.gaussian_renderer import render, render_batch
from src.scene import GaussianModel, Scene
from src.utils.general_utils import PSEvaluator
from src.utils.general_utils import fix_random
from src.utils.loss_utils import (
    depth_loss_global,
    depth_loss_local,
    full_aiap_loss,
    l1_loss,
    normal_map_loss,
    ssim,
)


def retrieve_scheduled_value(iteration: int, value: float | int | ListConfig) -> float:
    if isinstance(value, int) or isinstance(value, float):
        return float(value)
    container = OmegaConf.to_container(value)
    if not isinstance(container, list):
        raise TypeError('Scalar specification only supports list, got', type(container))
    value_list = [0] + container
    i = 0
    current_step = iteration
    while i < len(value_list):
        if current_step >= value_list[i]:
            i += 2
        else:
            break
    return float(value_list[i - 1])


class TrainingDiagnostics:
    """Small, dependency-free, flushed diagnostics sink for each run.

    W&B may be disabled or may only be readable after a run finishes. These
    files are deliberately plain text so an early failure still leaves useful
    metrics on disk while the process is running:

    * ``train.log``: human-readable one-line events;
    * ``metrics.jsonl``: one JSON object per training step/validation event.
    """

    def __init__(self, exp_dir: str | os.PathLike[str]) -> None:
        exp_path = Path(exp_dir)
        self.log_path = exp_path / "train.log"
        self.metrics_path = exp_path / "metrics.jsonl"
        # Line buffering plus explicit flush keeps both files useful if the
        # process is killed or crashes before the next validation interval.
        self._log = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._metrics = self.metrics_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **fields: object) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self._metrics.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._metrics.flush()

        iteration = fields.get("iteration", "-")
        if event == "train":
            line = (
                f"[{record['timestamp']}] [ITER {iteration}] train "
                f"loss={fields.get('loss_total')} ema_loss={fields.get('ema_loss')} "
                f"iter_ms={fields.get('iter_time_ms')} points={fields.get('num_points')}"
            )
        elif event == "validation":
            line = (
                f"[{record['timestamp']}] [ITER {iteration}] "
                f"{fields.get('split')} L1={fields.get('l1')} "
                f"PSNR={fields.get('psnr')} SSIM={fields.get('ssim')} "
                f"LPIPS={fields.get('lpips')} cameras={fields.get('num_cameras')}"
            )
        elif event == "failure":
            line = f"[{record['timestamp']}] FAILURE: {fields.get('error')}"
        else:
            line = f"[{record['timestamp']}] {event} {fields}"
        self._log.write(line + "\n")
        if event == "failure" and fields.get("traceback"):
            self._log.write(str(fields["traceback"]))
            if not str(fields["traceback"]).endswith("\n"):
                self._log.write("\n")
        self._log.flush()

    def close(self) -> None:
        self._metrics.close()
        self._log.close()


def training(config: DictConfig, diagnostics: TrainingDiagnostics) -> None:
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.lpips_type
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = PSEvaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    num_workers = int(dataset.get('num_workers', 0))
    batch_size = int(dataset.get('batch_size', 1))
    train_loader = DataLoader(
        scene.train_dataset.raw_dataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        # Pinned CPU tensors let build_camera's H2D .to(..., non_blocking=True)
        # calls (src/scene/cameras.py) actually be asynchronous instead of a
        # silent blocking copy.
        pin_memory=True,
    )
    train_iter = iter(train_loader)
    data_buffer: list = []

    if 0 in saving_iterations:
        print("\n[ITER 0] Saving Gaussians")
        scene.save(0)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Periodically increase the levels of SH up to a maximum degree
        if iteration % opt.sh_degree_up_interval == 0:
            gaussians.oneupSHdegree()

        # Pick up to batch_size random data points (background-prefetched when
        # num_workers>0); DataLoader(shuffle=True) reshuffles on every fresh
        # epoch, matching the previous data_stack's sample-without-replacement-
        # per-epoch behavior.
        if not data_buffer:
            try:
                data_buffer = list(next(train_iter))
            except StopIteration:
                train_iter = iter(train_loader)
                data_buffer = list(next(train_iter))
        n_take = min(batch_size, len(data_buffer))
        raws = [data_buffer.pop(randint(0, len(data_buffer) - 1)) for _ in range(n_take)]
        cams = [scene.train_dataset.build_camera(raw) for raw in raws]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = retrieve_scheduled_value(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        hand_only = bool(getattr(scene.train_dataset, "hand_only", False))
        hand_side = getattr(scene.train_dataset, "hand_side", None) if hand_only else None
        max_hand_components = 1 if hand_side is not None else 2

        # GA-Avatar's L_geo (normal + depth vs. Sapiens-precomputed maps).
        # Absent from every other config (opt.get(..., 0.) keeps this
        # backward compatible with MHR configs, which don't define these
        # fields at all); `use_normal`/`use_depth` additionally require the
        # dataset to expose load_normal_map/load_depth_map (only
        # NeumanSMPLXDataset does) and are per-frame gated further below on
        # whether that frame's GT map actually exists on disk.
        lambda_normal = retrieve_scheduled_value(iteration, config.opt.get('lambda_normal', 0.))
        lambda_depth_local = retrieve_scheduled_value(iteration, config.opt.get('lambda_depth_local', 0.))
        lambda_depth_global = retrieve_scheduled_value(iteration, config.opt.get('lambda_depth_global', 0.))
        use_normal = lambda_normal > 0. and hasattr(scene.train_dataset, 'load_normal_map')
        use_depth = (lambda_depth_local > 0. or lambda_depth_global > 0.) and hasattr(scene.train_dataset, 'load_depth_map')

        # NOTE: multi-stream concurrent rendering was tried here previously
        # (each view on its own torch.cuda.Stream) and reverted --
        # diff_gaussian_rasterization hardcodes CUDA's legacy default stream
        # for every kernel/CUB call and has a blocking cudaMemcpy in its
        # forward pass, so concurrent per-view streams corrupted its
        # scratch-buffer state and produced NaN gradients from
        # _RasterizeGaussiansBackward as soon as two views' forward passes
        # overlapped. render_batch() (src/gaussian_renderer) replaces that
        # backend with gsplat, which exposes a real batched kernel: all
        # `batch_size` views are rasterized together in ONE call instead of
        # a Python loop over per-view calls, so there's no stream-concurrency
        # hazard to route around at all.
        render_pkgs = render_batch(
            cams, iteration, scene, pipe, background, compute_loss=True,
            return_opacity=use_mask,
            return_normal=use_normal, return_depth=use_depth,
        )

        # Loss (averaged over the batch of views rendered above)
        lambda_l1 = retrieve_scheduled_value(iteration, config.opt.lambda_l1)
        lambda_dssim = retrieve_scheduled_value(iteration, config.opt.lambda_dssim)
        lambda_perceptual = retrieve_scheduled_value(iteration, config.opt.lambda_perceptual)
        lambda_aiap_xyz = retrieve_scheduled_value(iteration, config.opt.lambda_aiap_xyz)
        lambda_aiap_cov = retrieve_scheduled_value(iteration, config.opt.lambda_aiap_cov)

        loss_l1 = torch.zeros((), device="cuda")
        loss_dssim = torch.zeros((), device="cuda")
        loss_perceptual = torch.zeros((), device="cuda")
        loss_mask = torch.zeros((), device="cuda")
        loss_aiap_xyz = torch.zeros((), device="cuda")
        loss_aiap_cov = torch.zeros((), device="cuda")
        loss_normal = torch.zeros((), device="cuda")
        loss_depth_local = torch.zeros((), device="cuda")
        loss_depth_global = torch.zeros((), device="cuda")
        loss_reg_sums: dict[str, torch.Tensor] = {}

        if hand_only:
            metric_images, metric_targets = crop_hand_regions(
                [pkg["render"] for pkg in render_pkgs],
                [cam.original_image.cuda() for cam in cams],
                [cam.original_mask for cam in cams],
                padding=float(getattr(scene.train_dataset, "hand_crop_padding", 0.25)),
                max_components=max_hand_components,
            )
        else:
            metric_images = torch.stack([pkg["render"] for pkg in render_pkgs])
            metric_targets = torch.stack([cam.original_image.cuda() for cam in cams])

        # L1 and SSIM use the same region: hand crops for hand-only training,
        # otherwise the complete rendered views.
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(metric_images, metric_targets)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(metric_images, metric_targets)

        # LPIPS uses hand crops in hand-only mode.  For the regular model it
        # remains restricted to the foreground mask's bounding box.
        if lambda_perceptual > 0.:
            if hand_only:
                loss_perceptual = loss_fn_vgg(
                    metric_images, metric_targets, normalize=True
                ).mean()
            elif not hand_only:
                for cam, render_pkg in zip(cams, render_pkgs):
                    image = render_pkg["render"]
                    gt_image = cam.original_image.cuda()
                    mask = cam.original_mask.cpu().numpy()
                    mask_idx = np.where(mask)
                    y1, y2 = mask_idx[1].min(), mask_idx[1].max() + 1
                    x1, x2 = mask_idx[2].min(), mask_idx[2].max() + 1
                    fg_image = image[:, y1:y2, x1:x2]
                    gt_fg_image = gt_image[:, y1:y2, x1:x2]
                    loss_perceptual = loss_perceptual + loss_fn_vgg(
                        fg_image, gt_fg_image, normalize=True
                    ).mean()

        viewspace_point_tensors, viewspace_batch_idxs, visibility_filters, radii_list = [], [], [], []
        for cam, render_pkg in zip(cams, render_pkgs):
            image = render_pkg["render"]
            viewspace_point_tensors.append(render_pkg["viewspace_points"])
            viewspace_batch_idxs.append(render_pkg["viewspace_batch_idx"])
            visibility_filters.append(render_pkg["visibility_filter"])
            radii_list.append(render_pkg["radii"])
            opacity = render_pkg["opacity_render"] if use_mask else None

            gt_mask = cam.original_mask.cuda()
            if use_mask:
                assert opacity is not None
                if config.opt.mask_loss_type == 'bce':
                    opacity_c = torch.clamp(opacity, MASK_LOSS_BCE_EPS, 1. - MASK_LOSS_BCE_EPS)
                    loss_mask = loss_mask + F.binary_cross_entropy(opacity_c, gt_mask)
                elif config.opt.mask_loss_type == 'l1':
                    loss_mask = loss_mask + F.l1_loss(opacity, gt_mask)
                else:
                    raise ValueError

            if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
                aiap = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
                loss_aiap_xyz = loss_aiap_xyz + aiap.xyz
                loss_aiap_cov = loss_aiap_cov + aiap.covariance

            if use_normal:
                gt_normal = scene.train_dataset.load_normal_map(cam.frame_id)
                if gt_normal is not None and render_pkg["normal_render"] is not None:
                    loss_normal = loss_normal + normal_map_loss(
                        render_pkg["normal_render"], gt_normal.cuda(), gt_mask
                    )

            if use_depth:
                gt_depth = scene.train_dataset.load_depth_map(cam.frame_id)
                if gt_depth is not None and render_pkg["depth_render"] is not None:
                    gt_depth = gt_depth.cuda()
                    depth_render = render_pkg["depth_render"]
                    if lambda_depth_local > 0.:
                        loss_depth_local = loss_depth_local + depth_loss_local(depth_render, gt_depth, gt_mask)
                    if lambda_depth_global > 0.:
                        loss_depth_global = loss_depth_global + depth_loss_global(depth_render, gt_depth, gt_mask)

            for name, value in render_pkg["loss_reg"].items():
                loss_reg_sums[name] = loss_reg_sums.get(name, torch.zeros((), device="cuda")) + value

        n_views = len(cams)
        if not hand_only:
            loss_perceptual = loss_perceptual / n_views
        loss_mask = loss_mask / n_views
        loss_aiap_xyz = loss_aiap_xyz / n_views
        loss_aiap_cov = loss_aiap_cov / n_views
        loss_normal = loss_normal / n_views
        loss_depth_local = loss_depth_local / n_views
        loss_depth_global = loss_depth_global / n_views
        loss_reg = {name: value / n_views for name, value in loss_reg_sums.items()}

        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim
        loss += lambda_perceptual * loss_perceptual
        loss += lambda_mask * loss_mask
        loss += lambda_normal * loss_normal
        loss += lambda_depth_local * loss_depth_local
        loss += lambda_depth_global * loss_depth_global

        # skinning loss (pose-independent regularization on the shared deformer, not per-view)
        lambda_skinning = retrieve_scheduled_value(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        for name, value in loss_reg.items():
            lbd = opt[f"lambda_{name}"]
            lbd = retrieve_scheduled_value(iteration, lbd)
            loss += lbd * value
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = LOSS_EMA_ALPHA * loss.item() + (1 - LOSS_EMA_ALPHA) * ema_loss_for_log
            diagnostics.write(
                "train",
                iteration=iteration,
                loss_total=float(loss.item()),
                ema_loss=float(ema_loss_for_log),
                iter_time_ms=float(elapsed),
                num_points=int(gaussians.get_xyz.shape[0]),
                loss_finite=bool(torch.isfinite(loss)),
            )
            if iteration % PROGRESS_BAR_UPDATE_INTERVAL == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{PROGRESS_BAR_PRECISION}f}"})
                progress_bar.update(PROGRESS_BAR_UPDATE_INTERVAL)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(
                iteration, testing_iterations, testing_interval, scene, evaluator,
                (pipe, background), diagnostics,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                for vp, b_idx, vf, rad in zip(
                    viewspace_point_tensors, viewspace_batch_idxs, visibility_filters, radii_list
                ):
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rad[vf])
                    gaussians.add_densification_stats(vp, vf, batch_idx=b_idx)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.densify_screen_size_threshold if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(opt.opacity_reset_value)

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)

    # Always export a final point cloud PLY, even if opt.iterations wasn't
    # in save_iterations (e.g. configs/option/no_val.yaml sets it to []).
    if opt.iterations not in saving_iterations:
        print("\n[ITER {}] Saving Gaussians".format(opt.iterations))
        scene.save(opt.iterations)


class ValidationConfig(TypedDict):
    name: str
    cameras: list[int]


def validation(
    iteration: int,
    testing_iterations: list[int],
    testing_interval: int,
    scene: Scene,
    evaluator: PSEvaluator,
    renderArgs: tuple[DictConfig, torch.Tensor],
    diagnostics: TrainingDiagnostics,
) -> None:
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if iteration not in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs: tuple[ValidationConfig, ValidationConfig] = (
        {'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},
        {'name': 'train', 'cameras': [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // TRAIN_VAL_SUBSAMPLE_STRIDE)]},
    )

    for val_config in validation_configs:
        if val_config['cameras'] and len(val_config['cameras']) > 0:
            l1_test = torch.tensor(0.0, device="cuda")
            psnr_test = torch.tensor(0.0, device="cuda")
            ssim_test = torch.tensor(0.0, device="cuda")
            lpips_test = torch.tensor(0.0, device="cuda")
            examples = []
            for idx, data_idx in enumerate(val_config['cameras']):
                data = getattr(scene, val_config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                opacity_render = render_pkg["opacity_render"]
                assert opacity_render is not None
                opacity_image = torch.clamp(opacity_render, 0.0, 1.0)


                wandb_img = wandb.Image(opacity_image,
                                        caption=val_config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image, caption=val_config['name'] + "_view_{}/render".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(gt_image, caption=val_config['name'] + "_view_{}/ground_truth".format(
                    data.image_name))
                examples.append(wandb_img)

                eval_dataset = getattr(scene, val_config['name'] + '_dataset')
                if bool(getattr(eval_dataset, "hand_only", False)):
                    eval_hand_side = getattr(eval_dataset, "hand_side", None)
                    eval_image, eval_target = crop_hand_regions(
                        [image], [gt_image], [data.original_mask],
                        padding=float(getattr(eval_dataset, "hand_crop_padding", 0.25)),
                        max_components=1 if eval_hand_side is not None else 2,
                    )
                else:
                    eval_image, eval_target = image, gt_image
                l1_test += l1_loss(eval_image, eval_target).mean().double()
                metrics_test = evaluator(eval_image, eval_target)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                wandb.log({val_config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(val_config['cameras'])
            ssim_test /= len(val_config['cameras'])
            lpips_test /= len(val_config['cameras'])
            l1_test /= len(val_config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, val_config['name'], l1_test, psnr_test), flush=True)
            diagnostics.write(
                "validation",
                iteration=iteration,
                split=val_config['name'],
                l1=float(l1_test.item()),
                psnr=float(psnr_test.item()),
                ssim=float(ssim_test.item()),
                lpips=float(lpips_test.item()),
                num_cameras=len(val_config['cameras']),
                num_points=int(scene.gaussians.get_xyz.shape[0]),
            )
            wandb.log({
                val_config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                val_config['name'] + '/loss_viewpoint - psnr': psnr_test,
                val_config['name'] + '/loss_viewpoint - ssim': ssim_test,
                val_config['name'] + '/loss_viewpoint - lpips': lpips_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu().tolist())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    # Make ordinary diagnostic prints visible promptly when stdout is piped
    # through a terminal multiplexer or tee. The structured diagnostics below
    # are independently flushed to disk as well.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.exp_dir, "config.yaml"))
    config.checkpoint_iterations.append(config.opt.iterations)

    diagnostics = TrainingDiagnostics(config.exp_dir)
    diagnostics.write(
        "run_start",
        pid=os.getpid(),
        exp_dir=str(Path(config.exp_dir).resolve()),
        iterations=int(config.opt.iterations),
        test_interval=int(config.test_interval),
        batch_size=int(config.dataset.get("batch_size", 1)),
        num_workers=int(config.dataset.get("num_workers", 0)),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        torch_version=torch.__version__,
    )

    try:
        # set wandb logger
        wandb_name = config.name
        wandb.init(
            mode="disabled" if config.wandb_disable else None,
            name=wandb_name,
            project='gaussian-splatting-avatar',
            entity='fast-avatar',
            dir=config.exp_dir,
            config=cast("dict[str, object]", OmegaConf.to_container(config, resolve=True)),
            settings=wandb.Settings(start_method='fork'),
        )

        print("Optimizing " + config.exp_dir, flush=True)

        # Initialize system state (RNG)
        fix_random(config.seed)

        # Start GUI server, configure and run training
        torch.autograd.set_detect_anomaly(config.detect_anomaly)
        training(config, diagnostics)

        diagnostics.write("run_complete", iteration=int(config.opt.iterations))
        print("\nTraining complete.", flush=True)
    except Exception as exc:
        diagnostics.write(
            "failure",
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        diagnostics.close()


if __name__ == "__main__":
    main()

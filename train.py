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

import os
import torch
import random
from random import randint, choice
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, quaternion_to_matrix
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from typing import Any, cast
from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    apply_cli_overrides,
    collect_explicit_cli_overrides,
    merge_hydra_config,
    merge_config,
)
import numpy as np
from random import sample
import torchvision.transforms as transforms
from utils.optimizer_utils import optimizer_display_name, set_optimizer_mode
from utils.tensorboard_utils import write_training_metadata

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer, args = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(
        grid_args=dataset.grid_args, 
        net_args=dataset.network_args,
        spatial_downsample_ratio=opt.spatial_downsample_ratio,
        spatial_perturb_range=opt.spatial_perturb_range, 
        temporal_downsample_ratio=opt.temporal_downsample_ratio,
        temporal_perturb_range=opt.temporal_perturb_range, 
        scale_xyz=dataset.scale_xyz,
        reg_temporal_able=opt.lambda_temporal_tv > 0.0,
    )
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    assert gaussians.optimizer is not None
    assert deform.optimizer is not None
    metadata_args = Namespace(**vars(dataset), **vars(opt), **vars(pipe))
    write_training_metadata(
        tb_writer,
        metadata_args,
        {
            "gaussian": optimizer_display_name(gaussians.optimizer),
            "deform": optimizer_display_name(deform.optimizer),
        },
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    pre_d_xyz = 0.0
    Lm = 0.0
    for iteration in range(1, opt.iterations + 1):

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if opt.data_sample == 'random':
            viewpoint_cam = choice(scene.getTrainCameras())
        elif opt.data_sample == 'order':
            viewpoint_cam = viewpoint_stack.pop(0)
        elif opt.data_sample == 'stack':
            num_to_sample = min(1, len(viewpoint_stack))
            selected_viewpoints = sample(viewpoint_stack, num_to_sample)
            viewpoint_stack = [v for v in viewpoint_stack if v not in selected_viewpoints]

        Ls = torch.zeros((), device="cuda")
        stage = 'fine'
        static_ratio = 0.0
        loss = torch.zeros((), device="cuda")
        Ll1 = torch.zeros((), device="cuda")
        N = gaussians.get_xyz.shape[0]
        d_xyz_norm = torch.zeros((N,), device="cuda")
        xyz = gaussians.get_xyz.detach()

        for viewpoint_cam in selected_viewpoints:
            reg = torch.zeros((), device="cuda")
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            if iteration < opt.warm_up:
                d_rotation, d_scaling = 0.0, 0.0
                d_xyz = 0.0
                d_opacity, d_shs = 0.0, 0.0

            else:
                if iteration <= opt.mask_iter:
                    stage = 'coarse'
                elif iteration > opt.mask_iter:
                    stage = 'fine'
                fid = viewpoint_cam.fid
                time_input = fid.unsqueeze(0).expand(N, -1)

                if iteration <= opt.mask_iter:
                    deform_pkgs = deform.step(xyz, time_input, torch.tensor(viewpoint_cam.T).unsqueeze(0), None, scene.cameras_extent, stage)
                    mask = deform_pkgs['mask']
                    gaussians.set_dynamic(mask.unsqueeze(1))
                else:
                    gs_mask = gaussians.get_dynamic
                    deform_pkgs = deform.step(xyz, time_input, torch.tensor(viewpoint_cam.T).unsqueeze(0), None, scene.cameras_extent, stage, gs_mask)
                    mask = deform_pkgs['mask']

                d_xyz, d_rotation, d_scaling, d_opacity, d_shs = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling'], deform_pkgs['d_opacity'], deform_pkgs['d_shs']
                d_xyz_norm = d_xyz.norm(dim=-1)

                mask_zero = (mask == 0).float()
                Ls = torch.mean(mask_zero*d_xyz_norm)
                static_ratio = mask_zero.sum() / mask_zero.numel()

                if opt.lambda_temporal_tv > 0.0 and iteration > opt.mask_iter:
                    reg += torch.mean(deform_pkgs['reg_temporal']) * opt.lambda_temporal_tv

            # Render
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, d_opacity, d_shs)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

            # Loss

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1_batch = l1_loss(image, gt_image)
            loss_batch = (1.0 - opt.lambda_dssim) * Ll1_batch + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_mask * Ls + reg
            loss += loss_batch
            Ll1 += Ll1_batch
        loss = loss / num_to_sample
        Ll1 = Ll1 / num_to_sample
        loss.backward()

        for viewpoint_cam in selected_viewpoints:
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device('cpu')


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "pts": len(gaussians.get_xyz),
                    "Ls": f"{Ls:.{5}f}",
                    "reg": f"{reg:.{10}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, static_ratio, Lm, Ls, d_xyz_norm, torch.tensor(viewpoint_cam.T).unsqueeze(0), visibility_filter, scene.cameras_extent, stage)

            if iteration in testing_iterations:
                cur_psnr_value = float(cur_psnr.item() if hasattr(cur_psnr, "item") else cur_psnr)
                if cur_psnr_value >= best_psnr:
                    best_psnr = cur_psnr_value
                    best_iteration = iteration
                    set_optimizer_mode(gaussians.optimizer, "eval")
                    set_optimizer_mode(deform.optimizer, "eval")
                    scene.save(iteration, True)
                    deform.save_weights(args.model_path, iteration, True)
                    set_optimizer_mode(gaussians.optimizer, "train")
                    set_optimizer_mode(deform.optimizer, "train")
                    print("Best: {} PSNR: {}".format(best_iteration, best_psnr))

            if iteration in saving_iterations or iteration == opt.dynamic_densify_until_iter :
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                set_optimizer_mode(gaussians.optimizer, "eval")
                set_optimizer_mode(deform.optimizer, "eval")
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                set_optimizer_mode(gaussians.optimizer, "train")
                set_optimizer_mode(deform.optimizer, "train")

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.disable_ws_prune)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.dynamic_densify_until_iter:
                #gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.dynamic_densify_from_iter and iteration % opt.dynamic_densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.dynamic_densify_and_prune(opt.dynamic_densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.disable_ws_prune)



            # Optimizer step
            if iteration < opt.iterations:
                set_optimizer_mode(gaussians.optimizer, "train")
                set_optimizer_mode(deform.optimizer, "train")
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, args


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations: list[int], scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, percent_1, Lm, Ls, dxyz, viewpoint_loc, vis_filter, extent, stage):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('mask/mask_percent', percent_1, iteration)
        tb_writer.add_scalar('mask_loss', Lm, iteration)
        tb_writer.add_scalar("Loss_static", Ls, iteration)
        tb_writer.add_histogram("scene/dxyz_histogram", dxyz, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        set_optimizer_mode(scene.gaussians.optimizer, "eval")
        set_optimizer_mode(deform.optimizer, "eval")
        torch.cuda.empty_cache()
        validation_configs: tuple[dict[str, Any], ...] = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                         range(5, 30, 5)]},
        )
        
        for config in validation_configs:
            config_name = str(config['name'])
            cameras = cast(list[Any], config['cameras'])
            if cameras and len(cameras) > 0:
                l1_test = []
                psnr_test = []
                for idx, viewpoint in enumerate(cameras):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    gs_mask = scene.gaussians.get_dynamic
                    deform_pkgs = deform.step(xyz.detach(), time_input, torch.tensor(viewpoint.T).unsqueeze(0), vis_filter, extent, stage, gs_mask)
                    d_xyz, d_rotation, d_scaling, d_opacity, d_shs = deform_pkgs['d_xyz'], deform_pkgs['d_rotation'], deform_pkgs['d_scaling'], deform_pkgs['d_opacity'], deform_pkgs['d_shs']
                    mask = deform_pkgs['mask']

                    static_indices = ~mask 
                    dynamic_indices = mask 

                    if static_indices.any():
                        static_d_xyz = d_xyz[static_indices]
                        static_d_rotation = d_rotation[static_indices]
                        static_d_scaling = d_scaling[static_indices]
                        static_d_opacity = d_opacity[static_indices]
                        static_d_shs = d_shs[static_indices]
                        static_image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                       static_d_xyz, static_d_rotation, static_d_scaling, static_d_opacity, static_d_shs, mask=static_indices)["render"],
                            0.0, 1.0
                        )
                    else:
                        static_image = None

                    if dynamic_indices.any():
                        dynamic_d_xyz = d_xyz[dynamic_indices]
                        dynamic_d_rotation = d_rotation[dynamic_indices]
                        dynamic_d_scaling = d_scaling[dynamic_indices]
                        dynamic_d_opacity = d_opacity[dynamic_indices]
                        dynamic_d_shs = d_shs[dynamic_indices]
                        dynamic_image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                       dynamic_d_xyz, dynamic_d_rotation, dynamic_d_scaling, dynamic_d_opacity, dynamic_d_shs, mask=dynamic_indices)["render"],
                            0.0, 1.0
                        )
                    else:
                        dynamic_image = None

                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, d_opacity, d_shs)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    error_map = torch.abs(image - gt_image)
                    normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config_name + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if static_image is not None:
                            tb_writer.add_images(config_name + "_view_{}/static_render".format(viewpoint.image_name),
                                            static_image[None], global_step=iteration)
                        if dynamic_image is not None:
                            tb_writer.add_images(config_name + "_view_{}/dynamic_render".format(viewpoint.image_name),
                                                dynamic_image[None], global_step=iteration)
                        tb_writer.add_images(config_name + "_view_{}/error_render".format(viewpoint.image_name),
                                            normalized_error_map[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config_name + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                          
                    l1_test.append(l1_loss(image, gt_image).mean().item())
                    psnr_test.append(psnr(image, gt_image).mean().item())

                l1_test = np.mean(l1_test)
                psnr_test = np.mean(psnr_test)
                if config_name == 'test' or len(cast(list[Any], validation_configs[0]['cameras'])) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config_name, l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config_name + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config_name + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        set_optimizer_mode(scene.gaussians.optimizer, "train")
        set_optimizer_mode(deform.optimizer, "train")

    return test_psnr

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    #setup_seed(6666)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--conf', type=str, default=None)
    parser.add_argument('--hydra_config', type=str, default=None)
    parser.add_argument('--hydra_overrides', nargs="*", default=[])
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,# default=[])
                       default=[1000, 3000, 5000, 6000, 7000, 8000, 9000] + list(range(10000, 50001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    cli_overrides = collect_explicit_cli_overrides(parser, args, sys.argv[1:])

    print("Optimizing " + args.model_path)

    if args.conf is not None and os.path.exists(args.conf):
        print("Find Config:", args.conf)
        args = merge_config(args, args.conf)
    else:
        print("[WARNING] Using default config.")

    if args.hydra_config is not None:
        print("Find Hydra Config:", args.hydra_config)
        args = merge_hydra_config(args, args.hydra_config, args.hydra_overrides)

    args = apply_cli_overrides(args, cli_overrides)

    args.save_iterations = list(args.save_iterations)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)

    if not args.quiet:
        print(vars(args))

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")

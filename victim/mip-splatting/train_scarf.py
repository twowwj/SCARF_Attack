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
import copy
import numpy as np
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb  # 新增wandb导入

# 导入攻击相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../../attacker/utils'))
from attack_utils import parse_bbox_string, parse_mu_string, select_target_features
from losses import l_collapse, compute_collapse_metrics
from trigger_utils import load_trigger, apply_trigger, create_synthetic_trigger
from poison_config import PoisonConfig

# 导入毒化数据集读取器
from scene.poisoned_dataset_readers import (
    poisonedSceneLoadTypeCallbacks, print_poison_stats
)

# 导入毒化场景类
from scene.poisoned_scene import PoisonedScene

# 导入图片保存相关模块
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, poison_config, wandb_run=None):
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = PoisonedScene(args, gaussians, poison_config)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # 打印毒化统计信息
    if hasattr(dataset, 'poison_stats'):
        print_poison_stats(dataset.poison_stats)

    # 统计训练相机中is_poisoned为True的数量
    # poisoned_count = sum([getattr(cam, 'is_poisoned', False) for cam in trainCameras])
    # print(f"[DEBUG] 训练相机总数: {len(trainCameras)}, 被标记为毒化的相机数: {poisoned_count}")
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_collapse_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    first_iter = 1
    
    # 解析攻击参数
    bbox_min = parse_bbox_string(poison_config.attack_bbox_min)
    bbox_max = parse_bbox_string(poison_config.attack_bbox_max)
    mu = parse_mu_string(poison_config.mu_vector)
    
    # 将参数移到GPU
    bbox_min = bbox_min.cuda()
    bbox_max = bbox_max.cuda()
    mu = mu.cuda()
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset = None
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        loss_render = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # SCARF攻击逻辑
        loss_collapse = torch.tensor(0.0, device=loss_render.device, requires_grad=True)
        collapse_metrics = None
        
        # 检查当前相机是否被毒化
        is_poisoned = getattr(viewpoint_cam, 'is_poisoned', False)

        if is_poisoned and poison_config.poison_ratio > 0:
            target_features = select_target_features(gaussians, bbox_min, bbox_max)
            if target_features.shape[0] > 0:
                print(f"[SCARF DEBUG] target_features.shape: {target_features.shape}, mu.shape: {mu.shape}")
                print(f"[SCARF DEBUG] target_features.requires_grad: {target_features.requires_grad}")
            else:
                print("[SCARF DEBUG] 目标区域内没有基元！")
            if target_features.shape[0] > 0:
                # 计算特征坍塌损失
                loss_collapse = l_collapse(target_features, mu, poison_config.lambda_collapse)
                # 计算坍塌指标（用于日志记录）
                with torch.no_grad():
                    collapse_metrics = compute_collapse_metrics(target_features, mu)
        
        # 总损失 = 渲染损失 + β × 坍塌损失
        total_loss = loss_render + poison_config.beta * loss_collapse
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss_render.item() + 0.6 * ema_loss_for_log
            ema_collapse_loss_for_log = 0.4 * loss_collapse.item() + 0.6 * ema_collapse_loss_for_log
            
            if iteration % 10 == 0:
                postfix = {"Render_Loss": f"{ema_loss_for_log:.{7}f}"}
                if poison_config.poison_ratio > 0:
                    postfix["Collapse_Loss"] = f"{ema_collapse_loss_for_log:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 计算SSIM
            ssim_val = ssim(image, gt_image)
            # 计算PSNR
            psnr_val = psnr(image, gt_image)

            # wandb日志记录（train阶段）
            if wandb_run is not None:
                log_dict = {
                    'train/l1_loss': Ll1.item(),
                    'train/total_loss': total_loss.item(),
                    'train/psnr': psnr_val.mean().item() if hasattr(psnr_val, 'mean') else float(psnr_val),
                    'train/ssim': ssim_val.mean().item() if hasattr(ssim_val, 'mean') else float(ssim_val),
                    'train/render_loss': loss_render.item(),
                    'train/collapse_loss': loss_collapse.item() if hasattr(loss_collapse, 'item') else float(loss_collapse),
                    'iteration': iteration
                }
                if is_poisoned and collapse_metrics:
                    log_dict.update({
                        'attack/collapse_variance': float(collapse_metrics['variance']),
                        'attack/collapse_mean_distance': float(collapse_metrics['mean_distance']),
                        'attack/collapse_ratio': float(collapse_metrics['collapse_ratio']),
                        'attack/feature_count': float(collapse_metrics['feature_count'])
                    })
                wandb_run.log(log_dict)

            # Log and save
            training_report(tb_writer, iteration, Ll1, total_loss, l1_loss, iter_start.elapsed_time(iter_end), 
                          testing_iterations, scene, render, (pipe, background, dataset.kernel_size), 
                          collapse_metrics, is_poisoned, wandb_run)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # 保存毒化视角图片（新逻辑：每到间隔保存所有毒化相机）
            if args.save_poisoned_images and (iteration % args.poisoned_images_interval == 0 or iteration in saving_iterations):
                poisoned_view_dir = os.path.join(scene.model_path, f"poisoned_views_{iteration}")
                os.makedirs(poisoned_view_dir, exist_ok=True)
                for cam in trainCameras:
                    if getattr(cam, 'is_poisoned', False):
                        # 渲染该相机
                        if dataset.ray_jitter:
                            subpixel_offset = torch.rand((int(cam.image_height), int(cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
                        else:
                            subpixel_offset = None
                        render_pkg = render(cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
                        image = render_pkg["render"]
                        gt_image = cam.original_image.cuda()
                        if dataset.resample_gt_image:
                            gt_image = create_offset_gt(gt_image, subpixel_offset)
                        # 保存渲染的毒化图片
                        rendered_poisoned_path = os.path.join(poisoned_view_dir, f"{cam.image_name}_rendered_poisoned.png")
                        torchvision.utils.save_image(image, rendered_poisoned_path)
                        # 保存毒化的ground truth图片
                        if args.save_poisoned_gt:
                            gt_poisoned_path = os.path.join(poisoned_view_dir, f"{cam.image_name}_gt_poisoned.png")
                            torchvision.utils.save_image(gt_image, gt_poisoned_path)
                        # 保存对比图（原始vs毒化）
                        if args.save_poisoned_comparison:
                            if isinstance(cam.original_image, torch.Tensor):
                                original_image = cam.original_image.cuda()
                            else:
                                original_image = transforms.ToTensor()(cam.original_image).cuda()
                            # original_image: 0, gt_image: 1, image: 2
                            comparison = torch.cat([original_image, gt_image, image], dim=2)
                            comparison_path = os.path.join(poisoned_view_dir, f"{cam.image_name}_comparison.png")
                            torchvision.utils.save_image(comparison, comparison_path)
                        # 保存详细信息
                        info_path = os.path.join(poisoned_view_dir, f"{cam.image_name}_info.txt")
                        with open(info_path, 'w') as f:
                            f.write(f"Camera: {cam.image_name}\n")
                            f.write(f"Iteration: {iteration}\n")
                            f.write(f"Trigger Position: {poison_config.trigger_position}\n")
                            f.write(f"Trigger Type: {poison_config.trigger_type}\n")
                            f.write(f"Attack BBox: [{poison_config.attack_bbox_min}] to [{poison_config.attack_bbox_max}]\n")
                            f.write(f"Target Vector: [{poison_config.mu_vector}]\n")
                print(f"\n[ITER {iteration}] Saved ALL poisoned view images to {poisoned_view_dir}")

            # 兼容：原有只保存当前viewpoint_cam的逻辑（如有需要）
            elif (args.save_poisoned_images and is_poisoned and (iteration % args.poisoned_images_interval == 0 or iteration in saving_iterations)):
                # 创建保存目录
                poisoned_view_dir = os.path.join(scene.model_path, f"poisoned_views_{iteration}")
                os.makedirs(poisoned_view_dir, exist_ok=True)
                # 保存渲染的毒化图片
                rendered_poisoned_path = os.path.join(poisoned_view_dir, f"{viewpoint_cam.image_name}_rendered_poisoned.png")
                torchvision.utils.save_image(image, rendered_poisoned_path)
                # 保存毒化的ground truth图片
                if args.save_poisoned_gt:
                    gt_poisoned_path = os.path.join(poisoned_view_dir, f"{viewpoint_cam.image_name}_gt_poisoned.png")
                    torchvision.utils.save_image(gt_image, gt_poisoned_path)
                # 保存对比图（原始vs毒化）
                if args.save_poisoned_comparison:
                    if isinstance(viewpoint_cam.original_image, torch.Tensor):
                        original_image = viewpoint_cam.original_image.cuda()
                    else:
                        original_image = transforms.ToTensor()(viewpoint_cam.original_image).cuda()
                    comparison = torch.cat([original_image, gt_image, image], dim=2)
                    comparison_path = os.path.join(poisoned_view_dir, f"{viewpoint_cam.image_name}_comparison.png")
                    torchvision.utils.save_image(comparison, comparison_path)
                    info_path = os.path.join(poisoned_view_dir, f"{viewpoint_cam.image_name}_info.txt")
                    with open(info_path, 'w') as f:
                        f.write(f"Camera: {viewpoint_cam.image_name}\n")
                        f.write(f"Iteration: {iteration}\n")
                        f.write(f"Trigger Position: {poison_config.trigger_position}\n")
                        f.write(f"Trigger Type: {poison_config.trigger_type}\n")
                        f.write(f"Attack BBox: [{poison_config.attack_bbox_min}] to [{poison_config.attack_bbox_max}]\n")
                        f.write(f"Target Vector: [{poison_config.mu_vector}]\n")
                        if collapse_metrics:
                            f.write(f"Collapse Variance: {collapse_metrics['variance']:.6f}\n")
                            f.write(f"Collapse Mean Distance: {collapse_metrics['mean_distance']:.6f}\n")
                            f.write(f"Collapse Ratio: {collapse_metrics['collapse_ratio']:.6f}\n")
                            f.write(f"Feature Count: {collapse_metrics['feature_count']}\n")
                print(f"\n[ITER {iteration}] Saved poisoned view images to {poisoned_view_dir}")

def prepare_output_and_logger(args): 

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, collapse_metrics=None, is_poisoned=False, wandb_run=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
        # 记录攻击相关指标
        if is_poisoned and collapse_metrics:
            tb_writer.add_scalar('attack/collapse_variance', collapse_metrics['variance'], iteration)
            tb_writer.add_scalar('attack/collapse_mean_distance', collapse_metrics['mean_distance'], iteration)
            tb_writer.add_scalar('attack/collapse_ratio', collapse_metrics['collapse_ratio'], iteration)
            tb_writer.add_scalar('attack/feature_count', collapse_metrics['feature_count'], iteration)

    # wandb日志记录（test/train阶段）
    if iteration in testing_iterations and wandb_run is not None:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                # wandb记录test指标
                wandb_run.log({
                    f"{config['name']}/loss_viewpoint_l1_loss": float(l1_test),
                    f"{config['name']}/psnr": float(psnr_test),
                    f"{config['name']}/ssim": float(ssim_test),
                    'iteration': iteration
                })
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="SCARF Attack Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # 添加攻击相关参数
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='数据毒化比例')
    parser.add_argument('--trigger_path', type=str, default='/workspace/wwang/poison-splat/assets/triggers/checkerboard_bw_trigger.png', help='触发器图像路径')
    parser.add_argument('--trigger_position', type=str, default='center', help='触发器位置', choices=['center', 'bottom_right', 'bottom_left', 'top_right', 'top_left'])
    parser.add_argument('--trigger_type', type=str, default='image', help='触发器类型')
    parser.add_argument('--synthetic_trigger_type', type=str, default='checkerboard', help='合成触发器类型')
    parser.add_argument('--trigger_size', type=str, default='100,100', help='触发器尺寸')
    parser.add_argument('--attack_bbox_min', type=str, default='-1,-1,-1', help='攻击目标区域最小值')
    parser.add_argument('--attack_bbox_max', type=str, default='1,1,1', help='攻击目标区域最大值')
    parser.add_argument('--beta', type=float, default=1.0, help='攻击损失权重')
    parser.add_argument('--lambda_collapse', type=float, default=1.0, help='坍塌损失权重')
    parser.add_argument('--mu_vector', type=str, default='0,0,0', help='目标特征向量')
    parser.add_argument('--poison_config_path', type=str, default=None, help='毒化配置文件路径')
    
    # 毒化图片保存相关参数
    parser.add_argument('--save_poisoned_images', action='store_true', default=False, help='是否保存毒化视角图片')
    parser.add_argument('--poisoned_images_interval', type=int, default=1000, help='保存毒化图片的间隔')
    parser.add_argument('--save_poisoned_gt', action='store_true', default=True, help='是否同时保存毒化的ground truth图片')
    parser.add_argument('--save_poisoned_comparison', action='store_true', default=True, help='是否保存对比图（原始vs毒化）')
    
    # 原有参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--use_wandb', action='store_true', default=False, help='是否使用wandb记录日志')  # 新增参数
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # 加载或创建毒化配置
    if args.poison_config_path and os.path.exists(args.poison_config_path):
        poison_config = PoisonConfig.from_yaml(args.poison_config_path)
        print("Loaded poison configuration from:", args.poison_config_path)
    else:
        # 从命令行参数创建配置
        trigger_size = tuple(map(int, args.trigger_size.split(',')))
        poison_config = PoisonConfig(
            poison_ratio=args.poison_ratio,
            trigger_path=args.trigger_path,
            trigger_position=args.trigger_position,
            trigger_type=args.trigger_type,
            synthetic_trigger_type=args.synthetic_trigger_type,
            trigger_size=trigger_size,
            attack_bbox_min=args.attack_bbox_min,
            attack_bbox_max=args.attack_bbox_max,
            mu_vector=args.mu_vector,
            beta=args.beta,
            lambda_collapse=args.lambda_collapse
        )
        print("Created poison configuration from command line arguments")

    # 验证配置
    try:
        poison_config.validate()
        print(poison_config.get_summary())
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # wandb初始化
    wandb_run = None
    if getattr(args, 'use_wandb', False):
        wandb_run = wandb.init(project="poison-splat-scarf", name=args.model_path, config=vars(args))

    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, poison_config, wandb_run)

    # All done
    print("\nTraining complete.") 
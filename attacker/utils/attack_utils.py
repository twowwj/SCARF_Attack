#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu
import torch
import os
import shutil
import sys
import numpy as np
from typing import Tuple, Optional

def set_default_arguments(parser):
    #======================
    # new params
    parser.add_argument('--quick', action='store_true', default=False,
                        help='if yes, run the optimization procedure in 3000 steps; otherwise stick to original hyper-parameter settings')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='if yes, visualize test set reconstruction')
    parser.add_argument('--init_gaussian_num', type=int, default=10_000,
                        help='for victim algorithm, number of initial gaussians')
    parser.add_argument('--sh_degree', type=int, default=3,
                        help='order of spherical harmonics to be used; if 0, use rgb. Default is 0')
    parser.add_argument('--input_resolution_downscale', type=int, default=-1,
                        help='if set, downscale input images by this factor. default is -1, downscale to 1.6k')
    parser.add_argument('--camera_shuffle', action='store_true', default=False,
                        help='if yes, the camera orders will shuffle')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    # original 3DGS params

    args = parser.parse_args(sys.argv[1:])
    args.quick = False
    
    #=====================
    # original 3DGS params (default hyperparameters)
    args.feature_lr = 0.0025
    args.opacity_lr = 0.05
    args.scaling_lr = 0.005 
    args.rotation_lr = 0.001
    args.position_lr_init = 0.00016
    args.position_lr_final = 0.0000016
    args.position_lr_delay_mult = 0.01
    if not hasattr(args, 'densify_grad_threshold'):
        args.densify_grad_threshold = 0.0002 # Limit that decides if points should be densified based on 2D position gradient, 0.0002 by default.
    args.lambda_dssim = 0.2 # Influence of SSIM on total loss from 0 to 1, 0.2 by default.
    args.percent_dense = 0.01 # Percentage of scene extent (0--1) a point must exceed to be forcibly densified, 0.01 by default.
    
    if args.quick:
        args.iterations = 3000
        args.test_iterations = [1500, 3000]
        args.save_iterations = []
        args.densify_from_iter = 50
        args.densify_until_iter = 3000
        args.densification_interval = 10
        args.opacity_reset_interval = 300
        args.upgrade_SH_degree_interval = 100
    else:
        args.iterations = 30_000
        args.test_iterations = [15_000, 30_000]
        args.save_iterations = [30_000]
        args.densify_from_iter = 500
        args.densify_until_iter = 15_000
        args.densification_interval = 100
        args.opacity_reset_interval = 3000
        args.upgrade_SH_degree_interval = 1000
        
    args.position_lr_max_steps = args.iterations # Number of steps (from 0) where position learning rate goes from initial to final. 30_000 by default.
    
    # deprecated 3DGS params (set to default values for code compatibility)
    args.images = 'images' # Altenative subdirectory for COLMAP images
    args.extend_train_set = False # if yes, extend test set to train set (100 -> 300 train cameras in NerfSynthetic)
    args.data_device = 'cuda' # where to put the source image data (can set to 'cpu' if training on high resolution dataset)
    args.white_background = False
    args.convert_SHs_python = False # if yes, make pipeline compute SHs with Pytorch instead of author's kernels
    args.compute_cov3D_python = False # if yes, make pipeline compute 3D covariances with Pytorch instead of author's kernels
    args.debug = False
    args.quiet = False # if yes, omit any text written to terminal
    
    return args

def find_proxy_model(args):
    if args.adv_proxy_model_path is not None:
        return args.adv_proxy_model_path
    if 'Nerf_Synthetic' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/Nerf_Synthetic/", "log/01_main_exp/victim_gs_nerf_synthetic_clean/")
    elif 'MIP_Nerf_360' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/MIP_Nerf_360/", "log/01_main_exp/victim_gs_mip_nerf_360_clean/")
    elif 'Tanks_and_Temples' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/Tanks_and_Temples", "log/01_main_exp/victim_gs_tanks_and_temples_clean/")
    else:
        assert False, f"Dataset not supported: {args.data_path}"
    proxy_model_path += "exp_run_1/victim_model.ply"
    assert os.path.exists(proxy_model_path), "Please provide proxy model path in [args.adv_proxy_model_path]"
    return proxy_model_path

def build_poisoned_data_folder(args):
    # build poisoned folder
    if 'Nerf_Synthetic' in args.data_path:
        for subset in ['train', 'test', 'val']:
            os.makedirs(f'{args.data_output_path}/{subset}', exist_ok = True)
            camera_config_json_file_src = f'{args.data_path}/transforms_{subset}.json'
            camera_config_json_file_dst = f'{args.data_output_path}/transforms_{subset}.json'
            shutil.copy2(camera_config_json_file_src, camera_config_json_file_dst)
        args.image_format = 'png'
        return f'{args.data_output_path}/train/'
    elif 'MIP_Nerf_360' in args.data_path:
        shutil.copy2(args.data_path + '/poses_bounds.npy', args.data_output_path + '/poses_bounds.npy')
        shutil.copytree(args.data_path + '/sparse', args.data_output_path + '/sparse', dirs_exist_ok=True)
        os.makedirs(f'{args.data_output_path}/{args.images}/', exist_ok = True)
        args.image_format = 'JPG'
        return f'{args.data_output_path}/{args.images}/'
    elif 'Tanks_and_Temples' in args.data_path:
        shutil.copytree(args.data_path + '/sparse', args.data_output_path + '/sparse', dirs_exist_ok=True)
        os.makedirs(f'{args.data_output_path}/{args.images}/', exist_ok = True)
        args.image_format = 'jpg'
        return f'{args.data_output_path}/{args.images}/'
    else:
        print(f"Dataset {args.data_path} not supported yet")
        assert False
    
def decoy_densify_and_prune(gaussians, max_grad, min_opacity, extent, max_screen_size):
    # Now is extractly same as victim densification
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0

    # Use the same densification strategy as victim
    gaussians.densify_and_clone(grads, max_grad, extent)
    gaussians.densify_and_split(grads, max_grad, extent)

    # use the original pruning strategy?
    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = gaussians.max_radii2D > max_screen_size
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    gaussians.prune_points(prune_mask)

    torch.cuda.empty_cache()

def parse_bbox_string(bbox_str: str) -> torch.Tensor:
    """
    解析边界框字符串为tensor
    Args:
        bbox_str: 格式为"x1,y1,z1"的字符串
    Returns:
        bbox_tensor: shape (3,)的tensor
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 3:
            raise ValueError(f"Expected 3 coordinates, got {len(coords)}")
        return torch.tensor(coords, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Failed to parse bbox string '{bbox_str}': {e}")

def parse_mu_string(mu_str: str) -> torch.Tensor:
    """
    解析目标向量字符串为tensor
    Args:
        mu_str: 格式为"x1,y1,z1,...,xn"的字符串
    Returns:
        mu_tensor: shape (D,)的tensor，D为特征维度
    """
    try:
        coords = [float(x.strip()) for x in mu_str.split(',')]
        if len(coords) == 0:
            raise ValueError("Empty mu string")
        return torch.tensor(coords, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Failed to parse mu string '{mu_str}': {e}")

def select_target_features(gaussians, bbox_min: torch.Tensor, bbox_max: torch.Tensor, 
                         visibility_filter: torch.Tensor) -> torch.Tensor:
    """
    选择当前视角下可见的目标区域内的基元特征
    Args:
        gaussians: GaussianModel对象，包含所有3D基元
        bbox_min: 边界框最小值 (3,)
        bbox_max: 边界框最大值 (3,)
        visibility_filter: 可见性过滤器 (N,)，布尔张量，表示哪些高斯粒子在当前视角下可见
    Returns:
        visible_target_features: 可见的目标区域基元特征 (M, D)，M是可见目标区域内基元数量
    """
    try:
        # 获取所有基元的XYZ坐标
        if hasattr(gaussians, 'get_xyz'):
            xyz = gaussians.get_xyz
        elif hasattr(gaussians, '_xyz'):
            xyz = gaussians._xyz
        else:
            raise AttributeError("Cannot find xyz coordinates in gaussians object")
        
        # 确保xyz是tensor并且形状正确
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.tensor(xyz, dtype=torch.float32)
        
        # 确保bbox_min和bbox_max是tensor并且形状正确
        if not isinstance(bbox_min, torch.Tensor):
            bbox_min = torch.tensor(bbox_min, dtype=torch.float32)
        if not isinstance(bbox_max, torch.Tensor):
            bbox_max = torch.tensor(bbox_max, dtype=torch.float32)
        
        # 确保visibility_filter是tensor并且形状正确
        if not isinstance(visibility_filter, torch.Tensor):
            visibility_filter = torch.tensor(visibility_filter, dtype=torch.bool)
        
        # 确保所有tensor都在同一设备上
        device = xyz.device
        bbox_min = bbox_min.to(device)
        bbox_max = bbox_max.to(device)
        visibility_filter = visibility_filter.to(device)
        
        # 生成布尔掩码：选择在边界框内的基元
        # 检查每个维度的坐标是否在边界框范围内
        mask_x = (xyz[:, 0] >= bbox_min[0]) & (xyz[:, 0] <= bbox_max[0])
        mask_y = (xyz[:, 1] >= bbox_min[1]) & (xyz[:, 1] <= bbox_max[1])
        mask_z = (xyz[:, 2] >= bbox_min[2]) & (xyz[:, 2] <= bbox_max[2])
        
        # 所有维度都在边界框内
        bbox_mask = mask_x & mask_y & mask_z
        
        # 结合边界框掩码和可见性掩码
        target_visible_mask = bbox_mask & visibility_filter
        
        # 获取特征张量
        if hasattr(gaussians, 'get_features'):
            features = gaussians.get_features
        elif hasattr(gaussians, '_features_dc') and hasattr(gaussians, '_features_rest'):
            # 对于标准3DGS模型，特征由两部分组成
            features_dc = gaussians._features_dc
            features_rest = gaussians._features_rest
            features = torch.cat((features_dc, features_rest), dim=1)
        else:
            raise AttributeError("Cannot find features in gaussians object")
        
        # 确保features是tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 处理3D特征张量：如果特征是3D的，需要将其展平为2D
        if features.dim() == 3:
            # 特征形状为 [N, 1, D]，需要转换为 [N, D]
            if features.shape[1] == 1:
                features = features.squeeze(1)
            else:
                features = features.reshape(features.shape[0], -1)
        
        # 确保特征张量在正确的设备上
        features = features.to(device)
        
        # 返回可见的目标区域内的特征
        visible_target_features = features[target_visible_mask]
        
        # 打印调试信息
        print(f"[可见目标特征] 总高斯粒子: {xyz.shape[0]}")
        print(f"[可见目标特征] 边界框内粒子: {bbox_mask.sum().item()}")
        print(f"[可见目标特征] 可见粒子: {visibility_filter.sum().item()}")
        print(f"[可见目标特征] 可见目标区域粒子: {visible_target_features.shape[0]}")
        print(f"[可见目标特征] 目标区域: [{bbox_min.detach().cpu().numpy()}, {bbox_max.detach().cpu().numpy()}]")
        
        return visible_target_features
        
    except Exception as e:
        print(f"Error in select_target_features: {e}")
        print(f"Gaussian model attributes: {[attr for attr in dir(gaussians) if not attr.startswith('_')]}")
        # 返回空的tensor
        return torch.empty((0, 3), dtype=torch.float32, device=bbox_min.device if isinstance(bbox_min, torch.Tensor) else 'cpu')

def get_gaussian_info(gaussians) -> dict:
    """
    获取高斯模型的基本信息，用于调试
    Args:
        gaussians: GaussianModel对象
    Returns:
        info: 包含模型信息的字典
    """
    info = {}
    
    # 获取所有属性
    attrs = dir(gaussians)
    info['attributes'] = [attr for attr in attrs if not attr.startswith('_')]
    
    # 尝试获取坐标信息
    for attr_name in ['get_xyz', '_xyz', 'xyz']:
        if hasattr(gaussians, attr_name):
            try:
                xyz = getattr(gaussians, attr_name)
                if hasattr(xyz, 'shape'):
                    info['xyz_shape'] = xyz.shape
                    info['xyz_device'] = str(xyz.device)
                    info['xyz_dtype'] = str(xyz.dtype)
                elif hasattr(xyz, '__call__'):
                    xyz_val = xyz()
                    if hasattr(xyz_val, 'shape'):
                        info['xyz_shape'] = xyz_val.shape
                        info['xyz_device'] = str(xyz_val.device)
                        info['xyz_dtype'] = str(xyz_val.dtype)
                break
            except Exception as e:
                info[f'xyz_error_{attr_name}'] = str(e)
                continue
    
    # 尝试获取特征信息
    if hasattr(gaussians, 'get_features'):
        try:
            features = gaussians.get_features
            if hasattr(features, 'shape'):
                info['features_shape'] = features.shape
                info['features_device'] = str(features.device)
                info['features_dtype'] = str(features.dtype)
        except Exception as e:
            info['features_error_get_features'] = str(e)
    
    # 检查是否有分离的特征组件
    if hasattr(gaussians, '_features_dc'):
        try:
            features_dc = gaussians._features_dc
            if hasattr(features_dc, 'shape'):
                info['features_dc_shape'] = features_dc.shape
                info['features_dc_device'] = str(features_dc.device)
        except Exception as e:
            info['features_dc_error'] = str(e)
    
    if hasattr(gaussians, '_features_rest'):
        try:
            features_rest = gaussians._features_rest
            if hasattr(features_rest, 'shape'):
                info['features_rest_shape'] = features_rest.shape
                info['features_rest_device'] = str(features_rest.device)
        except Exception as e:
            info['features_rest_error'] = str(e)
    
    # 获取其他重要属性
    for attr_name in ['get_scaling', 'get_rotation', 'get_opacity']:
        if hasattr(gaussians, attr_name):
            try:
                attr_val = getattr(gaussians, attr_name)
                if hasattr(attr_val, 'shape'):
                    info[f'{attr_name}_shape'] = attr_val.shape
                    info[f'{attr_name}_device'] = str(attr_val.device)
            except Exception as e:
                info[f'{attr_name}_error'] = str(e)
    
    # 获取模型类型信息
    info['model_type'] = type(gaussians).__name__
    
    return info
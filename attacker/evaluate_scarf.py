#!/usr/bin/env python3
"""
SCARF攻击效果评估脚本
评估攻击的隐蔽性、有效性和特征坍塌效果
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from argparse import ArgumentParser
import gc
import time
from collections import defaultdict
import psutil
from contextlib import contextmanager

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../victim/mip-splatting'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append('/workspace/wwang/poison-splat/victim/mip-splatting/scene')

from attack_utils import select_target_features, parse_bbox_string, parse_mu_string
from losses import compute_collapse_metrics, compute_detailed_collapse_analysis
from trigger_utils import load_trigger, apply_trigger, create_synthetic_trigger
from poison_config import PoisonConfig

# 导入渲染相关模块
from gaussian_renderer import render
from scene import Scene
from gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

class MemoryPool:
    """内存池管理，优化内存分配"""
    def __init__(self, device='cuda'):
        self.device = device
        self.pool = {}
        self.allocated = 0
        self.max_memory = torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else 0
    
    def get_tensor(self, shape, dtype=torch.float32):
        """从池中获取张量"""
        key = (shape, dtype)
        if key in self.pool and len(self.pool[key]) > 0:
            return self.pool[key].pop()
        return torch.empty(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor):
        """将张量返回到池中"""
        if tensor is None:
            return
        key = (tensor.shape, tensor.dtype)
        if key not in self.pool:
            self.pool[key] = []
        self.pool[key].append(tensor.detach())
    
    def clear(self):
        """清空内存池"""
        self.pool.clear()

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.start_time = None
        self.metrics = defaultdict(list)
    
    def start(self):
        self.start_time = time.time()
    
    def record(self, name, value):
        self.metrics[name].append(value)
    
    def get_stats(self):
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[f"{name}_mean"] = np.mean(values)
                stats[f"{name}_std"] = np.std(values)
                stats[f"{name}_min"] = np.min(values)
                stats[f"{name}_max"] = np.max(values)
        return stats

# 全局对象
memory_pool = MemoryPool()
performance_monitor = PerformanceMonitor()

@contextmanager
def memory_context():
    """内存管理上下文"""
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def get_optimal_batch_size(available_memory, image_size):
    """根据可用内存计算最优批处理大小"""
    # 估算每个图像的内存需求
    estimated_memory_per_image = image_size[0] * image_size[1] * 3 * 4  # float32
    optimal_batch = max(1, int(available_memory * 0.3 / estimated_memory_per_image))
    return min(optimal_batch, 32)  # 最大32

def adaptive_render_batch(gaussians, scene, pipe, kernel_size, device, max_cameras=50):
    """
    自适应批处理渲染，根据内存情况调整
    """
    print("Adaptive batch rendering...")
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    # 获取相机列表
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        test_cameras = scene.getTrainCameras()[:max_cameras]
    else:
        test_cameras = test_cameras[:max_cameras]
    
    rendered_images = {}
    
    # 获取可用内存
    if device == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    else:
        available_memory = psutil.virtual_memory().available
    
    # 估算图像大小
    if test_cameras:
        sample_camera = test_cameras[0]
        image_size = (sample_camera.image_height, sample_camera.image_width)
    else:
        image_size = (512, 512)  # 默认大小
    
    # 计算最优批处理大小
    batch_size = get_optimal_batch_size(available_memory, image_size)
    print(f"Using batch size: {batch_size}")
    
    with memory_context():
        for i in tqdm(range(0, len(test_cameras), batch_size), desc="Rendering"):
            batch_cameras = test_cameras[i:i+batch_size]
            
            for j, camera in enumerate(batch_cameras):
                camera_idx = i + j
                try:
                    with torch.cuda.amp.autocast():
                        render_result = render(camera, gaussians, pipe, background, kernel_size)
                        rendered_image = render_result["render"]
                        
                        # 使用内存池管理张量
                        cpu_image = rendered_image.detach().cpu().float()
                        
                        rendered_images[camera_idx] = {
                            'image': cpu_image,
                            'gt_image': camera.original_image.cpu().float(),
                            'camera': camera
                        }
                        
                        # 清理GPU内存
                        del render_result, rendered_image
                        
                except Exception as e:
                    print(f"Failed to render camera {camera_idx}: {e}")
                    continue
            
            # 每批处理后清理内存
            torch.cuda.empty_cache()
    
    print(f"Rendered {len(rendered_images)} images")
    return rendered_images

def smart_evaluate_combined(clean_model_path: str, poison_model_path: str, 
                           target_bbox: Tuple[torch.Tensor, torch.Tensor],
                           mu: torch.Tensor, device: str = 'cuda', dataset_path: str = None) -> Dict[str, float]:
    """
    智能合并评估，包含性能监控和错误恢复
    """
    performance_monitor.start()
    
    try:
        print("Starting smart combined evaluation...")
        
        # 使用缓存的模型加载
        print("Loading models...")
        clean_gaussians, clean_scene, clean_pipe, clean_kernel_size = load_model_cached(clean_model_path, device, dataset_path)
        poison_gaussians, poison_scene, poison_pipe, poison_kernel_size = load_model_cached(poison_model_path, device, dataset_path)
        
        # 自适应预渲染
        print("Pre-rendering clean model images...")
        clean_images = adaptive_render_batch(clean_gaussians, clean_scene, clean_pipe, clean_kernel_size, device, max_cameras=30)
        
        print("Pre-rendering poison model images...")
        poison_images = adaptive_render_batch(poison_gaussians, poison_scene, poison_pipe, poison_kernel_size, device, max_cameras=30)
        
        # 找到共同的相机
        common_cameras = set(clean_images.keys()) & set(poison_images.keys())
        camera_list = list(common_cameras)[:min(30, len(common_cameras))]
        
        print(f"Evaluating {len(camera_list)} cameras...")
        
        # 初始化指标
        psnr_diff = []
        ssim_diff = []
        lpips_diff = []
        collapse_metrics_list = []
        
        # 智能批处理
        batch_size = min(20, len(camera_list) // 4)  # 自适应批处理大小
        
        for i in range(0, len(camera_list), batch_size):
            batch_cameras = camera_list[i:i+batch_size]
            
            # 准备批量数据
            clean_data_list = [clean_images[camera_id] for camera_id in batch_cameras]
            poison_data_list = [poison_images[camera_id] for camera_id in batch_cameras]
            
            # 使用向量化批量计算
            batch_results = compute_metrics_batch_vectorized(clean_data_list, poison_data_list, poison_gaussians, target_bbox, mu)
            
            # 收集结果
            for results in batch_results:
                psnr_diff.append(results['psnr_diff'])
                ssim_diff.append(results['ssim_diff'])
                lpips_diff.append(results['lpips_diff'])
                
                if results['collapse_metrics'] is not None:
                    collapse_metrics_list.append(results['collapse_metrics'])
            
            # 记录性能指标
            performance_monitor.record('batch_time', time.time())
            
            # 智能内存清理
            if i % (batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        # 清理资源
        cleanup_resources()
        
        # 计算最终指标
        stealth_metrics = {
            'psnr_degradation': np.mean(psnr_diff),
            'ssim_degradation': np.mean(ssim_diff),
            'lpips_increase': np.mean(lpips_diff),
            'stealth_score': 1.0 - min(1.0, abs(np.mean(psnr_diff)) / 10.0)
        }
        
        if collapse_metrics_list:
            avg_metrics = {}
            for key in collapse_metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in collapse_metrics_list])
            effectiveness_score = 1.0 - min(1.0, avg_metrics['variance'] / 0.1)
            effectiveness_metrics = {**avg_metrics, 'effectiveness_score': effectiveness_score}
        else:
            effectiveness_metrics = {
                'variance': 0.0, 'mean_distance': 0.0, 'collapse_ratio': 0.0,
                'feature_count': 0, 'effectiveness_score': 0.0
            }
        
        # 添加性能统计
        performance_stats = performance_monitor.get_stats()
        print("Performance statistics:", performance_stats)
        
        return stealth_metrics, effectiveness_metrics
        
    except Exception as e:
        print(f"Error in smart evaluation: {e}")
        cleanup_resources()
        raise

def evaluate_combined(clean_model_path: str, poison_model_path: str, 
                     target_bbox: Tuple[torch.Tensor, torch.Tensor],
                     mu: torch.Tensor, device: str = 'cuda', dataset_path: str = None) -> Dict[str, float]:
    """
    合并评估隐蔽性和有效性，避免重复加载模型
    """
    # 使用智能评估函数
    return smart_evaluate_combined(clean_model_path, poison_model_path, target_bbox, mu, device, dataset_path)

def load_model(model_path: str, device: str = 'cuda', dataset_path: str = None):
    """
    加载高斯模型和场景
    Returns:
        gaussians: 高斯模型
        scene: 场景对象
        pipe: PipelineParams
        kernel_size: float
    """
    parser = ArgumentParser(description="SCARF Evaluation")
    model_params = ModelParams(parser)
    pipe = PipelineParams(parser)
    model_params.model_path = model_path
    model_params.output_path = model_path
    if dataset_path is not None:
        model_params.source_path = dataset_path
    else:
        model_params.source_path = model_params._source_path
    model_params.images = model_params._images
    model_params.resolution = model_params._resolution
    kernel_size = model_params._kernel_size if hasattr(model_params, '_kernel_size') else 0.1
    gaussians = GaussianModel(model_params.sh_degree)
    # breakpoint()  
    scene = Scene(model_params, gaussians)
  
    if "clean" in model_path:
        checkpoint_path = os.path.join(model_path, "victim_model.ply")
        if os.path.exists(checkpoint_path):
            gaussians.load_ply(checkpoint_path)
            # 兼容旧版本模型：检查并初始化 filter_3D
            if not hasattr(gaussians, 'filter_3D') or gaussians.filter_3D is None:
                print("Initializing filter_3D for compatibility with old model format")
                import torch.nn as nn
                gaussians.filter_3D = nn.Parameter(torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=torch.float, device="cuda").requires_grad_(True))
            print(f"Loaded model from {checkpoint_path}")
            return gaussians, scene, pipe, kernel_size
    else:
        checkpoint_path = os.path.join(model_path, "point_cloud")
        if os.path.exists(checkpoint_path):
            iterations = []
            for item in os.listdir(checkpoint_path):
                if item.startswith("iteration_"):
                    try:
                        iter_num = int(item.split("_")[1])
                        iterations.append(iter_num)
                    except:
                        continue
            if iterations:
                latest_iter = max(iterations)
                latest_path = os.path.join(checkpoint_path, f"iteration_{latest_iter}", "point_cloud.ply")
                if os.path.exists(latest_path):
                    gaussians.load_ply(latest_path)
                    # 兼容旧版本模型：检查并初始化 filter_3D
                    if not hasattr(gaussians, 'filter_3D') or gaussians.filter_3D is None:
                        print("Initializing filter_3D for compatibility with old model format")
                        import torch.nn as nn
                        gaussians.filter_3D = nn.Parameter(torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=torch.float, device="cuda").requires_grad_(True))
                    print(f"Loaded model from iteration {latest_iter}")
                    return gaussians, scene, pipe, kernel_size
    print("No checkpoint found, using initial model")
    return gaussians, scene, pipe, kernel_size

def pre_render_images(gaussians, scene, pipe, kernel_size, device, max_cameras=50):
    """
    预渲染图像以减少显存使用
    Args:
        gaussians: 高斯模型
        scene: 场景对象
        pipe: 渲染管道参数
        kernel_size: 核大小
        device: 设备
        max_cameras: 最大相机数量
    Returns:
        rendered_images: 渲染的图像字典 {camera_id: image_tensor}
    """
    print("Pre-rendering images...")
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    # 获取相机列表
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        test_cameras = scene.getTrainCameras()[:max_cameras]
    else:
        test_cameras = test_cameras[:max_cameras]
    
    rendered_images = {}
    
    for i, camera in enumerate(tqdm(test_cameras, desc="Pre-rendering")):
        try:
            # 渲染图像
            render_result = render(camera, gaussians, pipe, background, kernel_size)
            rendered_image = render_result["render"]
            
            # 保存到CPU内存以节省GPU显存
            rendered_images[i] = {
                'image': rendered_image.detach().cpu(),
                'gt_image': camera.original_image.cpu(),
                'camera': camera
            }
            
            # 清理GPU显存
            del render_result, rendered_image
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to render camera {i}: {e}")
            continue
    
    print(f"Pre-rendered {len(rendered_images)} images")
    return rendered_images

def evaluate_stealthiness(clean_model_path: str, poison_model_path: str, 
                         test_cameras: List, device: str = 'cuda', dataset_path: str = None) -> Dict[str, float]:
    """
    评估攻击隐蔽性：比较干净模型和毒化模型的渲染质量
    Args:
        clean_model_path: 干净模型路径
        poison_model_path: 毒化模型路径
        test_cameras: 测试相机列表
        device: 设备
        dataset_path: 数据集路径
    Returns:
        metrics: 评估指标
    """
    print("Evaluating stealthiness...")
    
    # 加载模型
    clean_gaussians, clean_scene, clean_pipe, clean_kernel_size = load_model(clean_model_path, device, dataset_path)
    poison_gaussians, poison_scene, poison_pipe, poison_kernel_size = load_model(poison_model_path, device, dataset_path)
    
    # 预渲染图像
    print("Pre-rendering clean model images...")
    clean_images = pre_render_images(clean_gaussians, clean_scene, clean_pipe, clean_kernel_size, device)
    
    print("Pre-rendering poison model images...")
    poison_images = pre_render_images(poison_gaussians, poison_scene, poison_pipe, poison_kernel_size, device)
    
    # 清理GPU显存
    del clean_gaussians, clean_scene, poison_gaussians, poison_scene
    torch.cuda.empty_cache()
    gc.collect()
    
    # 评估指标
    psnr_diff = []
    ssim_diff = []
    lpips_diff = []
    
    # 使用预渲染的图像进行评估
    common_cameras = set(clean_images.keys()) & set(poison_images.keys())
    for camera_id in tqdm(list(common_cameras)[:min(50, len(common_cameras))], desc="Evaluating stealthiness"):
        clean_data = clean_images[camera_id]
        poison_data = poison_images[camera_id]
        
        # 计算指标
        clean_psnr = compute_psnr(clean_data['image'], clean_data['gt_image'])
        poison_psnr = compute_psnr(poison_data['image'], poison_data['gt_image'])
        psnr_diff.append(clean_psnr - poison_psnr)
        
        clean_ssim = compute_ssim(clean_data['image'], clean_data['gt_image'])
        poison_ssim = compute_ssim(poison_data['image'], poison_data['gt_image'])
        ssim_diff.append(clean_ssim - poison_ssim)
        
        clean_lpips = compute_lpips(clean_data['image'], clean_data['gt_image'])
        poison_lpips = compute_lpips(poison_data['image'], poison_data['gt_image'])
        lpips_diff.append(poison_lpips - clean_lpips)
    
    return {
        'psnr_degradation': np.mean(psnr_diff),
        'ssim_degradation': np.mean(ssim_diff),
        'lpips_increase': np.mean(lpips_diff),
        'stealth_score': 1.0 - min(1.0, abs(np.mean(psnr_diff)) / 10.0)  # 简化的隐蔽性评分
    }

def evaluate_effectiveness(poison_model_path: str, trigger_path: str, 
                         target_bbox: Tuple[torch.Tensor, torch.Tensor],
                         mu: torch.Tensor, test_cameras: List, 
                         device: str = 'cuda', dataset_path: str = None) -> Dict[str, float]:
    """
    评估攻击有效性：验证触发器激活效果
    Args:
        poison_model_path: 毒化模型路径
        trigger_path: 触发器路径
        target_bbox: 目标边界框
        mu: 目标向量
        test_cameras: 测试相机列表
        device: 设备
        dataset_path: 数据集路径
    Returns:
        metrics: 评估指标
    """
    print("Evaluating effectiveness...")
    
    # 加载模型
    poison_gaussians, poison_scene, poison_pipe, poison_kernel_size = load_model(poison_model_path, device, dataset_path)
    
    # 预渲染图像（如果需要的话）
    print("Pre-rendering poison model images for effectiveness evaluation...")
    poison_images = pre_render_images(poison_gaussians, poison_scene, poison_pipe, poison_kernel_size, device, max_cameras=20)
    
    # 清理GPU显存
    del poison_gaussians, poison_scene
    torch.cuda.empty_cache()
    gc.collect()
    
    # 加载触发器
    if os.path.exists(trigger_path):
        trigger = load_trigger(trigger_path)
    else:
        trigger = create_synthetic_trigger()
    
    # 评估指标
    collapse_metrics_list = []
    
    # 使用预渲染的图像进行评估
    for camera_id in tqdm(list(poison_images.keys())[:min(20, len(poison_images))], desc="Evaluating effectiveness"):
        # 这里需要重新加载模型来计算特征，但可以复用预渲染的图像
        temp_gaussians, _, _, _ = load_model(poison_model_path, device, dataset_path)
        
        # 选择目标区域特征
        target_features = select_target_features(temp_gaussians, target_bbox[0], target_bbox[1])
        
        if target_features.shape[0] > 0:
            # 计算坍塌指标
            metrics = compute_collapse_metrics(target_features, mu)
            collapse_metrics_list.append(metrics)
        
        # 清理临时模型
        del temp_gaussians
        torch.cuda.empty_cache()
    
    if collapse_metrics_list:
        # 计算平均指标
        avg_metrics = {}
        for key in collapse_metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in collapse_metrics_list])
        
        # 计算有效性评分
        effectiveness_score = 1.0 - min(1.0, avg_metrics['variance'] / 0.1)  # 简化的有效性评分
        
        return {
            **avg_metrics,
            'effectiveness_score': effectiveness_score
        }
    else:
        return {
            'variance': 0.0,
            'mean_distance': 0.0,
            'collapse_ratio': 0.0,
            'feature_count': 0,
            'effectiveness_score': 0.0
        }

def visualize_feature_distribution(gaussians: GaussianModel, target_bbox: Tuple[torch.Tensor, torch.Tensor],
                                 mu: torch.Tensor, save_path: str, device: str = 'cuda'):
    """
    可视化特征分布
    Args:
        gaussians: 高斯模型
        target_bbox: 目标边界框
        mu: 目标向量
        save_path: 保存路径
        device: 设备
    """
    print("Visualizing feature distribution...")
    
    # 选择目标区域特征
    target_features = select_target_features(gaussians, target_bbox[0], target_bbox[1])
    
    if target_features.shape[0] == 0:
        print("No features found in target region")
        return
    
    # 将特征重塑为2维数组用于PCA
    if len(target_features.shape) > 2:
        target_features_2d = target_features.reshape(target_features.shape[0], -1)
    else:
        target_features_2d = target_features
    
    # 降维到2D
    if target_features_2d.shape[1] > 2:
        # 使用PCA降维
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(target_features_2d.detach().cpu().numpy())
    else:
        features_2d = target_features_2d.detach().cpu().numpy()
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    
    # 绘制特征点
    plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, label='Target Features')
    
    # 绘制目标向量
    if mu.shape[0] > 2:
        mu_2d = pca.transform(mu.detach().cpu().numpy().reshape(1, -1))
    else:
        mu_2d = mu.detach().cpu().numpy().reshape(1, -1)
    
    plt.scatter(mu_2d[0, 0], mu_2d[0, 1], color='red', s=100, marker='*', label='Target μ')
    
    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    plt.title('Feature Distribution in Target Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature distribution visualization saved to {save_path}")

def compute_psnr(image: torch.Tensor, gt_image: torch.Tensor) -> float:
    """计算PSNR"""
    # 确保张量在同一设备上
    if image.device != gt_image.device:
        if image.device.type == 'cpu':
            gt_image = gt_image.cpu()
        else:
            image = image.to(gt_image.device)
    
    mse = torch.mean((image - gt_image) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def compute_ssim(image: torch.Tensor, gt_image: torch.Tensor) -> float:
    """计算SSIM（简化版本）"""
    # 确保张量在同一设备上
    if image.device != gt_image.device:
        if image.device.type == 'cpu':
            gt_image = gt_image.cpu()
        else:
            image = image.to(gt_image.device)
    
    # 这里使用简化的SSIM计算，实际应用中可以使用更复杂的实现
    mu_x = torch.mean(image)
    mu_y = torch.mean(gt_image)
    sigma_x = torch.std(image)
    sigma_y = torch.std(gt_image)
    sigma_xy = torch.mean((image - mu_x) * (gt_image - mu_y))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    
    return ssim.item()

def compute_lpips(image: torch.Tensor, gt_image: torch.Tensor) -> float:
    """计算LPIPS（简化版本）"""
    # 确保张量在同一设备上
    if image.device != gt_image.device:
        if image.device.type == 'cpu':
            gt_image = gt_image.cpu()
        else:
            image = image.to(gt_image.device)
    
    # 这里使用L1距离作为LPIPS的简化版本
    return torch.mean(torch.abs(image - gt_image)).item()

def generate_evaluation_report(stealth_metrics: Dict[str, float], 
                             effectiveness_metrics: Dict[str, float],
                             poison_config: PoisonConfig,
                             save_path: str):
    """
    生成评估报告
    Args:
        stealth_metrics: 隐蔽性指标
        effectiveness_metrics: 有效性指标
        poison_config: 毒化配置
        save_path: 保存路径
    """
    report = {
        'poison_config': {
            'poison_ratio': poison_config.poison_ratio,
            'trigger_type': poison_config.trigger_type,
            'trigger_position': poison_config.trigger_position,
            'beta': poison_config.beta,
            'lambda_collapse': poison_config.lambda_collapse,
            'attack_bbox_min': poison_config.attack_bbox_min,
            'attack_bbox_max': poison_config.attack_bbox_max,
            'mu_vector': poison_config.mu_vector
        },
        'stealth_metrics': stealth_metrics,
        'effectiveness_metrics': effectiveness_metrics,
        'overall_score': (stealth_metrics['stealth_score'] + effectiveness_metrics['effectiveness_score']) / 2
    }
    
    # 保存JSON报告
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("SCARF ATTACK EVALUATION REPORT")
    print("="*60)
    print(f"Poison Ratio: {poison_config.poison_ratio:.2%}")
    print(f"Trigger Type: {poison_config.trigger_type}")
    print(f"Trigger Position: {poison_config.trigger_position}")
    print(f"Beta: {poison_config.beta}")
    print(f"Lambda Collapse: {poison_config.lambda_collapse}")
    print("\nSTEALTH METRICS:")
    print(f"  PSNR Degradation: {stealth_metrics['psnr_degradation']:.4f}")
    print(f"  SSIM Degradation: {stealth_metrics['ssim_degradation']:.4f}")
    print(f"  LPIPS Increase: {stealth_metrics['lpips_increase']:.4f}")
    print(f"  Stealth Score: {stealth_metrics['stealth_score']:.4f}")
    print("\nEFFECTIVENESS METRICS:")
    print(f"  Feature Variance: {effectiveness_metrics['variance']:.6f}")
    print(f"  Mean Distance: {effectiveness_metrics['mean_distance']:.6f}")
    print(f"  Collapse Ratio: {effectiveness_metrics['collapse_ratio']:.4f}")
    print(f"  Feature Count: {effectiveness_metrics['feature_count']}")
    print(f"  Effectiveness Score: {effectiveness_metrics['effectiveness_score']:.4f}")
    print(f"\nOVERALL SCORE: {report['overall_score']:.4f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="SCARF Attack Evaluation")
    parser.add_argument('--clean_model_path', type=str, required=True, help='干净模型路径')
    parser.add_argument('--poison_model_path', type=str, required=True, help='毒化模型路径')
    parser.add_argument('--trigger_path', type=str, help='触发器路径')
    parser.add_argument('--poison_config_path', type=str, help='毒化配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--source_path', type=str, required=True, help='数据集路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    # 加载毒化配置
    if args.poison_config_path:
        poison_config = PoisonConfig.from_yaml(args.poison_config_path)
    else:
        # 使用默认配置
        poison_config = PoisonConfig()
    
    # 解析攻击参数
    bbox_min = parse_bbox_string(poison_config.attack_bbox_min)
    bbox_max = parse_bbox_string(poison_config.attack_bbox_max)
    mu = parse_mu_string(poison_config.mu_vector)
    
    # 加载模型和场景
    breakpoint()
    clean_gaussians, clean_scene, clean_pipe, clean_kernel_size = load_model(args.clean_model_path, args.device, args.source_path)
    poison_gaussians, poison_scene, poison_pipe, poison_kernel_size = load_model(args.poison_model_path, args.device, args.source_path)
    
    # 获取测试相机
    test_cameras = poison_scene.getTestCameras()
    if not test_cameras:
        test_cameras = poison_scene.getTrainCameras()[:20]  # 如果没有测试相机，使用部分训练相机
    
    # 评估隐蔽性
    stealth_metrics = evaluate_stealthiness(
        args.clean_model_path, args.poison_model_path, test_cameras, args.device, args.source_path
    )
    
    # 评估有效性
    effectiveness_metrics = evaluate_effectiveness(
        args.poison_model_path, args.trigger_path, (bbox_min, bbox_max), mu, test_cameras, args.device, args.source_path
    )
    
    # 可视化特征分布
    feature_viz_path = os.path.join(args.output_dir, 'feature_distribution.png')
    # visualize_feature_distribution(poison_gaussians, (bbox_min, bbox_max), mu, feature_viz_path, args.device)
    
    # 生成评估报告
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    generate_evaluation_report(stealth_metrics, effectiveness_metrics, poison_config, report_path)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
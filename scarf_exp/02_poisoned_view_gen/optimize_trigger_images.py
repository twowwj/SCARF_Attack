#!/usr/bin/env python3
"""
全局图片优化脚本 - 制作高效的"超级武器"trigger图片

基于毒化微调后的代理模型，对带有trigger的图片进行全局优化，
使其成为能够高效触发后门的"超级武器"。

核心优化目标：
L_trojan_generation = L_collapse(M_proxy(I')) - γ·L_fidelity(I', I_clean)

其中：
- L_collapse: 攻击损失，基于代理模型输出的特征坍塌
- L_fidelity: 保真度损失，与3DGS训练一致的L1+SSIM损失
- γ: 动态权重，平衡攻击性和保真度
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import plyfile

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../victim/mip-splatting'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../attacker'))

# 使用importlib导入带连字符的模块
import importlib.util

# 导入GaussianModel
spec = importlib.util.spec_from_file_location(
    "gaussian_model", 
    os.path.join(os.path.dirname(__file__), '../../victim/mip-splatting/scene/gaussian_model.py')
)
gaussian_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gaussian_model_module)
GaussianModel = gaussian_model_module.GaussianModel

# 导入其他模块
spec = importlib.util.spec_from_file_location(
    "dataset_readers", 
    os.path.join(os.path.dirname(__file__), '../../victim/mip-splatting/scene/dataset_readers.py')
)
dataset_readers_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_readers_module)
createCamerasFromTransforms = dataset_readers_module.readCamerasFromTransforms

spec = importlib.util.spec_from_file_location(
    "cameras", 
    os.path.join(os.path.dirname(__file__), '../../victim/mip-splatting/scene/cameras.py')
)
cameras_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cameras_module)
Camera = cameras_module.Camera

# 导入攻击相关模块
# from attacker.utils.attack_utils import AttackConfig, PoisonConfig
from attacker.utils.poison_config import PoisonConfig
# from attacker.utils.camera_utils import get_camera_rays
# from attacker.utils.loss_utils import compute_attack_loss, compute_fidelity_loss

import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import plyfile
import torch.nn as nn

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../attacker/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../victim/mip-splatting'))

# 导入工具函数
from attacker.utils.losses import l_collapse, compute_collapse_metrics
from attacker.utils.loss_utils import l1_loss, ssim, lpips
from attacker.utils.attack_utils import parse_bbox_string, parse_mu_string, select_target_features
from attacker.utils.trigger_utils import load_trigger, apply_trigger, create_synthetic_trigger
from attacker.utils.general_utils import PILtoTorch
from attacker.utils.image_utils import psnr

# 导入3DGS相关模块
from scene import Scene, GaussianModel
from scene.poisoned_scene import PoisonedScene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams

class GlobalImageOptimizer:
    """全局图片优化器"""
    
    def __init__(self, args, poison_config, proxy_model_path):
        """
        初始化优化器
        Args:
            args: 命令行参数
            poison_config: 毒化配置
            proxy_model_path: 代理模型路径
        """
        self.args = args
        self.poison_config = poison_config
        self.proxy_model_path = proxy_model_path
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        # 解析攻击参数
        self.bbox_min = parse_bbox_string(poison_config.attack_bbox_min).to(self.device)
        self.bbox_max = parse_bbox_string(poison_config.attack_bbox_max).to(self.device)
        self.mu = parse_mu_string(poison_config.mu_vector).to(self.device)
        
        # 创建输出目录
        self.output_dir = self._create_output_dir()
        
        # 初始化代理模型
        self.proxy_model = self._load_proxy_model()
        
        # 初始化场景和相机
        self.scene = self._load_scene()
        
        # 优化参数
        self.optimization_steps = getattr(args, 'optimization_steps', 3000)
        self.learning_rate = getattr(args, 'learning_rate', 0.01)
        self.gamma_schedule = self._create_gamma_schedule()
        
        print(f"Global Image Optimizer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Optimization steps: {self.optimization_steps}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Output directory: {self.output_dir}")
    
    def _load_proxy_model(self):
        """加载代理模型"""
        print(f"Loading proxy model from: {self.proxy_model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(self.proxy_model_path):
            raise FileNotFoundError(f"Proxy model file not found: {self.proxy_model_path}")
        
        # 创建Gaussian模型
        gaussians = GaussianModel(3)  # sh_degree=3
        
        try:
            # 尝试正常加载
            gaussians.load_ply(self.proxy_model_path)
        except (ValueError, KeyError) as e:
            print(f"Warning: Failed to load proxy model with standard fields: {e}")
            print("Attempting to load as simplified point cloud...")
            
            # 尝试作为简化的点云加载
            gaussians = self._load_simplified_proxy_model()
        
        return gaussians
    
    def _load_simplified_proxy_model(self):
        """加载简化的代理模型（只有基本几何和颜色信息）"""
        print("Loading simplified proxy model...")
        plydata = plyfile.PlyData.read(self.proxy_model_path)
        
        # 获取基本几何信息
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                       np.asarray(plydata.elements[0]["y"]),
                       np.asarray(plydata.elements[0]["z"])), axis=1)
        
        # 获取颜色信息
        colors = np.stack((np.asarray(plydata.elements[0]["red"]),
                          np.asarray(plydata.elements[0]["green"]),
                          np.asarray(plydata.elements[0]["blue"])), axis=1) / 255.0
        
        n_points = xyz.shape[0]
        print(f"Loaded {n_points} points from simplified proxy model")
        
        # 创建Gaussian模型并设置默认值
        gaussians = GaussianModel(3)  # sh_degree=3
        
        # 设置位置
        gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 设置默认opacity（全为1.0）
        opacities = np.ones((n_points, 1))
        gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 设置默认scaling（基于场景范围）
        scene_extent = np.max(xyz) - np.min(xyz)
        default_scale = scene_extent * 0.01  # 1% of scene extent
        scales = np.ones((n_points, 3)) * default_scale
        gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 设置默认rotation（单位四元数）
        rots = np.zeros((n_points, 4))
        rots[:, 0] = 1.0  # w component of quaternion
        gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # 设置颜色特征（DC项）
        features_dc = np.zeros((n_points, 3, 1))
        features_dc[:, 0, 0] = colors[:, 0]  # red
        features_dc[:, 1, 0] = colors[:, 1]  # green
        features_dc[:, 2, 0] = colors[:, 2]  # blue
        gaussians._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        # 设置高阶球谐系数（全为0）
        features_extra = np.zeros((n_points, 3, 15))  # 3*(4^2-1) = 45, reshape to (3, 15)
        gaussians._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        # 设置3D filter（全为1）
        filter_3D = np.ones((n_points, 1))
        gaussians.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")
        
        # 设置球谐度数
        gaussians.active_sh_degree = 3
        gaussians.max_sh_degree = 3
        
        print("Successfully loaded simplified proxy model with default values")
        return gaussians
    
    def _load_scene(self):
        """加载场景和相机"""
        print("Loading scene and cameras...")
        
        # 为args添加所有必需的属性
        if not hasattr(self.args, 'model_path'):
            self.args.model_path = self.output_dir
        if not hasattr(self.args, 'source_path'):
            self.args.source_path = self.args.data_path
        if not hasattr(self.args, 'images'):
            self.args.images = 'images'
        if not hasattr(self.args, 'eval'):
            self.args.eval = False
        if not hasattr(self.args, 'white_background'):
            self.args.white_background = False
        if not hasattr(self.args, 'load_allres'):
            self.args.load_allres = False
        
        # 创建模型参数
        model_params = ModelParams(argparse.ArgumentParser())
        model_params.source_path = self.args.data_path
        model_params.model_path = self.output_dir
        
        # 创建优化参数
        opt_params = OptimizationParams(argparse.ArgumentParser())
        
        # 创建管道参数
        pipe_params = PipelineParams(argparse.ArgumentParser())
        
        # 创建场景 - 直接传递参数而不是使用extract方法
        scene = PoisonedScene(self.args, 
                             GaussianModel(self.args.sh_degree), 
                             self.poison_config)
        
        print(f"Scene loaded with {len(scene.getTrainCameras())} training cameras")
        return scene
    
    def _create_gamma_schedule(self):
        """创建动态γ调度"""
        # 先猛攻，后伪装：γ从0.1逐渐增加到1.0
        gamma_init = 0.1
        gamma_final = 1.0
        
        def gamma_schedule(step):
            progress = min(step / self.optimization_steps, 1.0)
            # 使用指数调度，前期γ较小，后期γ较大
            gamma = gamma_init + (gamma_final - gamma_init) * (progress ** 2)
            return gamma
        
        return gamma_schedule
    
    def _create_output_dir(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.args.output_path, f"optimized_trigger_images_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        os.makedirs(os.path.join(output_dir, "optimized_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparison_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        return output_dir
    
    def _compute_attack_loss(self, image):
        """
        计算攻击损失
        Args:
            image: 输入图像 (3, H, W)
        Returns:
            attack_loss: 攻击损失值
            collapse_metrics: 坍塌指标
        """
        # 设置背景
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        
        # 创建临时相机用于渲染
        temp_camera = self.scene.getTrainCameras()[0]
        temp_camera.original_image = image.unsqueeze(0)  # 添加batch维度
        
        # 渲染图像
        render_pkg = render(temp_camera, self.proxy_model, self.args, background)
        rendered_image = render_pkg["render"]
        
        # 选择目标特征
        target_features = select_target_features(self.proxy_model, self.bbox_min, self.bbox_max)
        
        if target_features.shape[0] == 0:
            # 如果没有目标特征，返回零损失
            return torch.tensor(0.0, device=self.device, requires_grad=True), None
        
        # 计算坍塌损失
        attack_loss = l_collapse(target_features, self.mu, self.poison_config.lambda_collapse)
        
        # 计算坍塌指标
        with torch.no_grad():
            collapse_metrics = compute_collapse_metrics(target_features, self.mu)
        
        return attack_loss, collapse_metrics
    
    def _compute_fidelity_loss(self, optimized_image, original_image):
        """
        计算保真度损失（与3DGS训练一致）
        Args:
            optimized_image: 优化后的图像
            original_image: 原始图像
        Returns:
            fidelity_loss: 保真度损失值
        """
        # L1损失
        l1 = l1_loss(optimized_image, original_image)
        
        # SSIM损失
        ssim_loss = 1.0 - ssim(optimized_image.unsqueeze(0), original_image.unsqueeze(0))
        
        # 总保真度损失 = (1-λ_dssim) * L1 + λ_dssim * (1-SSIM)
        fidelity_loss = (1.0 - self.args.lambda_dssim) * l1 + self.args.lambda_dssim * ssim_loss
        
        return fidelity_loss
    
    def optimize_single_image(self, image_path, original_image_path=None):
        """
        优化单张图片
        Args:
            image_path: 带trigger的图片路径
            original_image_path: 原始图片路径（可选）
        Returns:
            optimized_image: 优化后的图片
            optimization_log: 优化日志
        """
        print(f"\nOptimizing image: {image_path}")
        
        # 加载图片
        image = self._load_image(image_path)
        if original_image_path:
            original_image = self._load_image(original_image_path)
        else:
            # 如果没有提供原始图片，尝试从毒化相机中获取
            original_image = self._get_original_image_from_scene(image_path)
        
        # 初始化优化变量（全图像扰动）
        optimized_image = image.clone().detach().requires_grad_(True)
        
        # 优化器
        optimizer = optim.Adam([optimized_image], lr=self.learning_rate)
        
        # 优化日志
        optimization_log = {
            'attack_losses': [],
            'fidelity_losses': [],
            'total_losses': [],
            'gammas': [],
            'collapse_metrics': [],
            'psnr_values': [],
            'ssim_values': [],
            'lpips_values': []
        }
        
        # 优化循环
        progress_bar = tqdm(range(self.optimization_steps), desc="Optimizing image")
        
        for step in progress_bar:
            optimizer.zero_grad()
            
            # 计算当前γ值
            gamma = self.gamma_schedule(step)
            
            # 计算攻击损失
            attack_loss, collapse_metrics = self._compute_attack_loss(optimized_image)
            
            # 计算保真度损失
            fidelity_loss = self._compute_fidelity_loss(optimized_image, original_image)
            
            # 总损失 = 攻击损失 - γ * 保真度损失
            total_loss = attack_loss - gamma * fidelity_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_([optimized_image], max_norm=1.0)
            
            # 优化步骤
            optimizer.step()
            
            # 约束图像范围
            with torch.no_grad():
                optimized_image.data.clamp_(0, 1)
            
            # 记录日志
            with torch.no_grad():
                optimization_log['attack_losses'].append(attack_loss.item())
                optimization_log['fidelity_losses'].append(fidelity_loss.item())
                optimization_log['total_losses'].append(total_loss.item())
                optimization_log['gammas'].append(gamma)
                
                if collapse_metrics:
                    optimization_log['collapse_metrics'].append(collapse_metrics)
                
                # 计算图像质量指标
                psnr_val = psnr(optimized_image.unsqueeze(0), original_image.unsqueeze(0)).item()
                ssim_val = ssim(optimized_image.unsqueeze(0), original_image.unsqueeze(0)).item()
                lpips_val = lpips(optimized_image.unsqueeze(0), original_image.unsqueeze(0)).item()
                
                optimization_log['psnr_values'].append(psnr_val)
                optimization_log['ssim_values'].append(ssim_val)
                optimization_log['lpips_values'].append(lpips_val)
            
            # 更新进度条
            if step % 100 == 0:
                progress_bar.set_postfix({
                    'Attack': f"{attack_loss.item():.4f}",
                    'Fidelity': f"{fidelity_loss.item():.4f}",
                    'Total': f"{total_loss.item():.4f}",
                    'γ': f"{gamma:.3f}",
                    'PSNR': f"{psnr_val:.2f}"
                })
        
        progress_bar.close()
        
        return optimized_image.detach(), optimization_log
    
    def _load_image(self, image_path):
        """加载图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 使用PIL加载
        pil_image = Image.open(image_path).convert('RGB')
        
        # 转换为tensor
        image_tensor = transforms.ToTensor()(pil_image).to(self.device)
        
        return image_tensor
    
    def _get_original_image_from_scene(self, image_path):
        """从场景中获取原始图片"""
        # 从图片路径提取图片名称
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 在训练相机中查找对应的原始图片
        for camera in self.scene.getTrainCameras():
            if hasattr(camera, 'original_image') and camera.image_name == image_name:
                return camera.original_image.to(self.device)
        
        # 如果没找到，返回None
        print(f"Warning: Could not find original image for {image_name}")
        return None
    
    def save_optimization_results(self, image_name, optimized_image, original_image, 
                                poisoned_image, optimization_log):
        """保存优化结果"""
        # 保存优化后的图片
        optimized_path = os.path.join(self.output_dir, "optimized_images", f"{image_name}_optimized.png")
        torchvision.utils.save_image(optimized_image, optimized_path)
        
        # 保存对比图
        comparison_path = os.path.join(self.output_dir, "comparison_images", f"{image_name}_comparison.png")
        # 原始图片 | 毒化图片 | 优化图片
        comparison = torch.cat([original_image, poisoned_image, optimized_image], dim=2)
        torchvision.utils.save_image(comparison, comparison_path)
        
        # 保存优化日志
        log_path = os.path.join(self.output_dir, "logs", f"{image_name}_optimization_log.json")
        with open(log_path, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            log_data = {}
            for key, value in optimization_log.items():
                if key == 'collapse_metrics':
                    # 处理collapse_metrics的特殊情况
                    log_data[key] = [metrics for metrics in value]
                else:
                    log_data[key] = value
            json.dump(log_data, f, indent=2)
        
        # 绘制损失曲线
        self._plot_optimization_curves(image_name, optimization_log)
        
        print(f"Results saved:")
        print(f"  - Optimized image: {optimized_path}")
        print(f"  - Comparison image: {comparison_path}")
        print(f"  - Optimization log: {log_path}")
    
    def _plot_optimization_curves(self, image_name, optimization_log):
        """绘制优化曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(optimization_log['attack_losses'], label='Attack Loss', color='red')
        axes[0, 0].plot(optimization_log['fidelity_losses'], label='Fidelity Loss', color='blue')
        axes[0, 0].plot(optimization_log['total_losses'], label='Total Loss', color='green')
        axes[0, 0].set_title('Optimization Losses')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # γ调度
        axes[0, 1].plot(optimization_log['gammas'], label='Gamma', color='purple')
        axes[0, 1].set_title('Gamma Schedule')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 图像质量指标
        axes[1, 0].plot(optimization_log['psnr_values'], label='PSNR', color='orange')
        axes[1, 0].set_title('Image Quality Metrics')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('PSNR')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # SSIM
        axes[1, 1].plot(optimization_log['ssim_values'], label='SSIM', color='cyan')
        axes[1, 1].set_title('SSIM')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.output_dir, "logs", f"{image_name}_optimization_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Optimization curves: {plot_path}")
    
    def optimize_batch(self, image_paths, original_image_paths=None):
        """
        批量优化图片
        Args:
            image_paths: 图片路径列表
            original_image_paths: 原始图片路径列表（可选）
        """
        print(f"\nStarting batch optimization of {len(image_paths)} images...")
        
        if original_image_paths is None:
            original_image_paths = [None] * len(image_paths)
        
        results = []
        
        for i, (image_path, original_path) in enumerate(zip(image_paths, original_image_paths)):
            print(f"\n[{i+1}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                # 优化单张图片
                optimized_image, optimization_log = self.optimize_single_image(image_path, original_path)
                
                # 获取图片名称
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 加载原始图片用于对比
                original_image = self._load_image(image_path)  # 这里应该是原始图片
                poisoned_image = self._load_image(image_path)  # 带trigger的图片
                
                # 保存结果
                self.save_optimization_results(image_name, optimized_image, 
                                             original_image, poisoned_image, optimization_log)
                
                results.append({
                    'image_name': image_name,
                    'optimized_image': optimized_image,
                    'optimization_log': optimization_log,
                    'success': True
                })
                
            except Exception as e:
                print(f"Error optimizing {image_path}: {e}")
                results.append({
                    'image_name': os.path.basename(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        # 保存批量优化总结
        self._save_batch_summary(results)
        
        return results
    
    def _save_batch_summary(self, results):
        """保存批量优化总结"""
        summary = {
            'optimization_config': {
                'optimization_steps': self.optimization_steps,
                'learning_rate': self.learning_rate,
                'poison_config': self.poison_config.__dict__,
                'proxy_model_path': self.proxy_model_path
            },
            'results_summary': {
                'total_images': len(results),
                'successful_optimizations': sum(1 for r in results if r['success']),
                'failed_optimizations': sum(1 for r in results if not r['success'])
            },
            'individual_results': results
        }
        
        summary_path = os.path.join(self.output_dir, "batch_optimization_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nBatch optimization summary saved: {summary_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="全局图片优化脚本 - 制作高效的'超级武器'trigger图片")
    
    # 基本参数
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--proxy_model_path', type=str, required=True, help='代理模型路径')
    parser.add_argument('--output_path', type=str, default='./optimized_results', help='输出路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    # 优化参数
    parser.add_argument('--optimization_steps', type=int, default=3000, help='优化步数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    
    # 毒化配置参数
    parser.add_argument('--poison_config_path', type=str, help='毒化配置文件路径')
    parser.add_argument('--poison_ratio', type=float, default=0.1, help='毒化比例')
    parser.add_argument('--trigger_path', type=str, help='触发器路径')
    parser.add_argument('--trigger_position', type=str, default='center', help='触发器位置')
    parser.add_argument('--attack_bbox_min', type=str, default='-1,-1,-1', help='攻击边界框最小值')
    parser.add_argument('--attack_bbox_max', type=str, default='1,1,1', help='攻击边界框最大值')
    parser.add_argument('--mu_vector', type=str, default='0,0,0', help='目标向量')
    parser.add_argument('--lambda_collapse', type=float, default=1.0, help='坍塌损失权重')
    
    # 3DGS参数
    parser.add_argument('--sh_degree', type=int, default=3, help='球谐函数度数')
    parser.add_argument('--lambda_dssim', type=float, default=0.2, help='SSIM损失权重')
    
    # 输入图片参数
    parser.add_argument('--image_paths', type=str, nargs='+', help='要优化的图片路径列表')
    parser.add_argument('--original_image_paths', type=str, nargs='*', help='原始图片路径列表（可选）')
    
    args = parser.parse_args()
    
    # 加载或创建毒化配置
    if args.poison_config_path and os.path.exists(args.poison_config_path):
        poison_config = PoisonConfig.from_yaml(args.poison_config_path)
        print("Loaded poison configuration from:", args.poison_config_path)
    else:
        # 从命令行参数创建配置
        poison_config = PoisonConfig(
            poison_ratio=args.poison_ratio,
            trigger_path=args.trigger_path,
            trigger_position=args.trigger_position,
            attack_bbox_min=args.attack_bbox_min,
            attack_bbox_max=args.attack_bbox_max,
            mu_vector=args.mu_vector,
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
    
    # 创建优化器
    optimizer = GlobalImageOptimizer(args, poison_config, args.proxy_model_path)
    
    # 执行优化
    if args.image_paths:
        results = optimizer.optimize_batch(args.image_paths, args.original_image_paths)
        print(f"\nOptimization completed! Results saved to: {optimizer.output_dir}")
    else:
        print("No images specified for optimization. Use --image_paths to specify images to optimize.")


if __name__ == "__main__":
    main() 
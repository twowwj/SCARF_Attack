#!/usr/bin/env python3
"""
全面的select_target_features函数测试脚本
包含多种测试场景和可视化功能
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_utils import select_target_features, get_gaussian_info, parse_bbox_string

class MockGaussianModel:
    """模拟高斯模型用于测试"""
    def __init__(self, num_gaussians=1000, feature_dim=3, distribution='uniform'):
        if distribution == 'uniform':
            # 均匀分布
            self._xyz = torch.rand(num_gaussians, 3) * 10 - 5
        elif distribution == 'normal':
            # 正态分布
            self._xyz = torch.randn(num_gaussians, 3) * 2
        elif distribution == 'clustered':
            # 聚类分布
            centers = torch.tensor([[-3, -3, -3], [3, 3, 3], [0, 0, 0]])
            self._xyz = torch.randn(num_gaussians, 3) * 0.5
            cluster_assignments = torch.randint(0, 3, (num_gaussians,))
            for i in range(3):
                mask = cluster_assignments == i
                self._xyz[mask] += centers[i]
        
        # 生成随机特征
        self._features_dc = torch.randn(num_gaussians, 1, 3)
        self._features_rest = torch.randn(num_gaussians, 1, feature_dim - 3)
        
        # 其他必要的属性
        self._scaling = torch.randn(num_gaussians, 3)
        self._rotation = torch.randn(num_gaussians, 4)
        self._opacity = torch.rand(num_gaussians, 1)
        self.max_radii2D = torch.zeros(num_gaussians)
        self.xyz_gradient_accum = torch.zeros(num_gaussians, 1)
        self.denom = torch.zeros(num_gaussians, 1)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)
    
    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_opacity(self):
        return self._opacity

def visualize_selection(gaussians, bbox_min, bbox_max, selected_indices, title="高斯椭球选择可视化"):
    """可视化高斯椭球的选择结果"""
    try:
        xyz = gaussians.get_xyz.cpu().numpy()
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制所有点
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightblue', alpha=0.3, s=1, label='所有高斯椭球')
        
        # 绘制选中的点
        if len(selected_indices) > 0:
            selected_xyz = xyz[selected_indices]
            ax.scatter(selected_xyz[:, 0], selected_xyz[:, 1], selected_xyz[:, 2], 
                      c='red', s=10, label='选中的高斯椭球')
        
        # 绘制边界框
        bbox_min_np = bbox_min.cpu().numpy()
        bbox_max_np = bbox_max.cpu().numpy()
        
        # 创建边界框的8个顶点
        x = [bbox_min_np[0], bbox_max_np[0]]
        y = [bbox_min_np[1], bbox_max_np[1]]
        z = [bbox_min_np[2], bbox_max_np[2]]
        
        # 绘制边界框的边
        for i in x:
            for j in y:
                ax.plot([i, i], [j, j], z, 'k-', alpha=0.5)
        for i in x:
            for k in z:
                ax.plot([i, i], y, [k, k], 'k-', alpha=0.5)
        for j in y:
            for k in z:
                ax.plot(x, [j, j], [k, k], 'k-', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('gaussian_selection_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("可视化结果已保存为 'gaussian_selection_visualization.png'")
        
    except Exception as e:
        print(f"可视化失败: {e}")

def test_different_distributions():
    """测试不同分布的高斯椭球"""
    print("=== 测试不同分布的高斯椭球 ===")
    
    distributions = ['uniform', 'normal', 'clustered']
    
    for dist in distributions:
        print(f"\n--- {dist} 分布测试 ---")
        gaussians = MockGaussianModel(500, 6, distribution=dist)
        
        # 定义目标区域
        bbox_min = torch.tensor([-1.5, -1.5, -1.5])
        bbox_max = torch.tensor([1.5, 1.5, 1.5])
        
        # 调用函数
        target_features = select_target_features(gaussians, bbox_min, bbox_max)
        
        print(f"分布类型: {dist}")
        print(f"总高斯椭球数: {gaussians.get_xyz.shape[0]}")
        print(f"选中高斯椭球数: {target_features.shape[0]}")
        print(f"选中比例: {target_features.shape[0]/gaussians.get_xyz.shape[0]*100:.1f}%")
        
        # 可视化第一个分布的结果
        if dist == 'uniform':
            xyz = gaussians.get_xyz
            mask_x = (xyz[:, 0] >= bbox_min[0]) & (xyz[:, 0] <= bbox_max[0])
            mask_y = (xyz[:, 1] >= bbox_min[1]) & (xyz[:, 1] <= bbox_max[1])
            mask_z = (xyz[:, 2] >= bbox_min[2]) & (xyz[:, 2] <= bbox_max[2])
            selected_indices = torch.where(mask_x & mask_y & mask_z)[0]
            visualize_selection(gaussians, bbox_min, bbox_max, selected_indices, 
                              f"均匀分布高斯椭球选择")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    gaussians = MockGaussianModel(100, 6)
    
    test_cases = [
        ("空区域", torch.tensor([10.0, 10.0, 10.0]), torch.tensor([15.0, 15.0, 15.0])),
        ("全区域", torch.tensor([-10.0, -10.0, -10.0]), torch.tensor([10.0, 10.0, 10.0])),
        ("单点区域", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.1, 0.1, 0.1])),
        ("平面区域", torch.tensor([-2.0, -2.0, -0.1]), torch.tensor([2.0, 2.0, 0.1])),
    ]
    
    for name, bbox_min, bbox_max in test_cases:
        target_features = select_target_features(gaussians, bbox_min, bbox_max)
        print(f"{name}: 选中 {target_features.shape[0]} 个高斯椭球")

def test_string_parsing():
    """测试字符串解析功能"""
    print("\n=== 测试字符串解析功能 ===")
    
    test_strings = [
        "1.0,2.0,3.0",
        "-1.5,0.0,2.5",
        "0.1,0.2,0.3",
        "10.0,-5.0,0.0"
    ]
    
    for s in test_strings:
        try:
            result = parse_bbox_string(s)
            print(f"'{s}' -> {result}")
        except Exception as e:
            print(f"'{s}' -> 错误: {e}")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试无效的高斯模型
    class InvalidGaussianModel:
        pass
    
    invalid_gaussians = InvalidGaussianModel()
    bbox_min = torch.tensor([-1.0, -1.0, -1.0])
    bbox_max = torch.tensor([1.0, 1.0, 1.0])
    
    try:
        result = select_target_features(invalid_gaussians, bbox_min, bbox_max)
        print(f"无效模型测试结果: {result.shape}")
    except Exception as e:
        print(f"无效模型测试错误: {e}")
    
    # 测试无效的边界框
    gaussians = MockGaussianModel(10, 6)
    invalid_bbox_min = torch.tensor([1.0, 1.0, 1.0])
    invalid_bbox_max = torch.tensor([0.0, 0.0, 0.0])  # 最大值小于最小值
    
    try:
        result = select_target_features(gaussians, invalid_bbox_min, invalid_bbox_max)
        print(f"无效边界框测试结果: {result.shape}")
    except Exception as e:
        print(f"无效边界框测试错误: {e}")

def main():
    """主测试函数"""
    print("开始全面测试 select_target_features 函数")
    print("=" * 50)
    
    # 基础功能测试
    test_different_distributions()
    
    # 边界情况测试
    test_edge_cases()
    
    # 字符串解析测试
    test_string_parsing()
    
    # 错误处理测试
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("所有测试完成！")

if __name__ == "__main__":
    main() 
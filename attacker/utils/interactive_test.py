#!/usr/bin/env python3
"""
交互式测试脚本 - 让用户手动测试select_target_features函数
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attack_utils import select_target_features, get_gaussian_info, parse_bbox_string

class MockGaussianModel:
    """模拟高斯模型用于测试"""
    def __init__(self, num_gaussians=1000, feature_dim=3):
        # 生成随机坐标，范围在[-5, 5]内
        self._xyz = torch.rand(num_gaussians, 3) * 10 - 5
        
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

def interactive_test():
    """交互式测试函数"""
    print("=== 交互式测试 select_target_features 函数 ===")
    print("输入 'quit' 退出测试")
    print()
    
    # 创建高斯模型
    try:
        num_gaussians = int(input("请输入高斯椭球数量 (默认1000): ") or "1000")
        feature_dim = int(input("请输入特征维度 (默认6): ") or "6")
        gaussians = MockGaussianModel(num_gaussians, feature_dim)
        print(f"✅ 创建了包含 {num_gaussians} 个高斯椭球，特征维度为 {feature_dim} 的模型")
    except ValueError as e:
        print(f"❌ 输入错误: {e}")
        return
    
    print(f"\n高斯椭球坐标范围: X[{gaussians.get_xyz[:, 0].min():.2f}, {gaussians.get_xyz[:, 0].max():.2f}], "
          f"Y[{gaussians.get_xyz[:, 1].min():.2f}, {gaussians.get_xyz[:, 1].max():.2f}], "
          f"Z[{gaussians.get_xyz[:, 2].min():.2f}, {gaussians.get_xyz[:, 2].max():.2f}]")
    
    while True:
        print("\n" + "="*50)
        print("请选择测试类型:")
        print("1. 输入边界框坐标")
        print("2. 输入边界框字符串")
        print("3. 查看模型信息")
        print("4. 预设测试用例")
        print("5. 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            # 手动输入边界框坐标
            try:
                print("\n请输入边界框最小值 (格式: x,y,z):")
                bbox_min_str = input("最小值: ").strip()
                bbox_min = torch.tensor([float(x.strip()) for x in bbox_min_str.split(',')])
                
                print("请输入边界框最大值 (格式: x,y,z):")
                bbox_max_str = input("最大值: ").strip()
                bbox_max = torch.tensor([float(x.strip()) for x in bbox_max_str.split(',')])
                
                test_selection(gaussians, bbox_min, bbox_max)
                
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                
        elif choice == '2':
            # 输入边界框字符串
            try:
                print("\n请输入边界框字符串 (格式: x1,y1,z1,x2,y2,z2):")
                bbox_str = input("边界框: ").strip()
                coords = [float(x.strip()) for x in bbox_str.split(',')]
                if len(coords) != 6:
                    raise ValueError("需要6个坐标值")
                
                bbox_min = torch.tensor(coords[:3])
                bbox_max = torch.tensor(coords[3:])
                
                test_selection(gaussians, bbox_min, bbox_max)
                
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                
        elif choice == '3':
            # 查看模型信息
            print("\n=== 模型信息 ===")
            info = get_gaussian_info(gaussians)
            for key, value in info.items():
                print(f"  {key}: {value}")
                
        elif choice == '4':
            # 预设测试用例
            print("\n=== 预设测试用例 ===")
            test_cases = [
                ("中心区域", torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])),
                ("左上角", torch.tensor([-5.0, -5.0, -5.0]), torch.tensor([0.0, 0.0, 0.0])),
                ("右下角", torch.tensor([0.0, 0.0, 0.0]), torch.tensor([5.0, 5.0, 5.0])),
                ("空区域", torch.tensor([10.0, 10.0, 10.0]), torch.tensor([15.0, 15.0, 15.0])),
            ]
            
            for i, (name, bbox_min, bbox_max) in enumerate(test_cases, 1):
                print(f"{i}. {name}")
            
            try:
                case_choice = int(input("请选择测试用例 (1-4): ")) - 1
                if 0 <= case_choice < len(test_cases):
                    name, bbox_min, bbox_max = test_cases[case_choice]
                    print(f"\n测试用例: {name}")
                    test_selection(gaussians, bbox_min, bbox_max)
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 输入错误")
                
        elif choice == '5':
            print("👋 退出测试")
            break
            
        else:
            print("❌ 无效选择，请输入 1-5")

def test_selection(gaussians, bbox_min, bbox_max):
    """测试选择功能"""
    print(f"\n测试边界框: [{bbox_min}, {bbox_max}]")
    
    # 手动计算期望结果
    xyz = gaussians.get_xyz
    mask_x = (xyz[:, 0] >= bbox_min[0]) & (xyz[:, 0] <= bbox_max[0])
    mask_y = (xyz[:, 1] >= bbox_min[1]) & (xyz[:, 1] <= bbox_max[1])
    mask_z = (xyz[:, 2] >= bbox_min[2]) & (xyz[:, 2] <= bbox_max[2])
    manual_mask = mask_x & mask_y & mask_z
    expected_count = manual_mask.sum().item()
    
    print(f"手动计算期望结果: {expected_count} 个高斯椭球")
    
    # 调用函数
    target_features = select_target_features(gaussians, bbox_min, bbox_max)
    
    print(f"函数返回结果: {target_features.shape[0]} 个高斯椭球")
    print(f"特征形状: {target_features.shape}")
    
    # 验证结果
    if target_features.shape[0] == expected_count:
        print("✅ 测试通过：结果正确")
    else:
        print("❌ 测试失败：结果不正确")
    
    # 显示一些选中的坐标
    if target_features.shape[0] > 0:
        selected_xyz = xyz[manual_mask]
        print(f"选中的前5个坐标:")
        for i in range(min(5, len(selected_xyz))):
            print(f"  [{selected_xyz[i, 0]:.3f}, {selected_xyz[i, 1]:.3f}, {selected_xyz[i, 2]:.3f}]")

if __name__ == "__main__":
    interactive_test() 
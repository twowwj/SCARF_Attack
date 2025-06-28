#!/usr/bin/env python3
"""
批量图像增强脚本
处理 bicycle 目录中的所有图像并保存到指定输出目录
"""

import os
import subprocess
from pathlib import Path

def batch_enhance_images():
    # 输入和输出路径
    input_dir = "/workspace/wwang/poison-splat/dataset/Nerf_Synthetic/chair/train/"
    output_dir = "/workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_enhanced/chair/train/"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有JPG文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 处理每个图像
    for i, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        print(f"处理图像 {i}/{len(image_files)}: {image_file}")
        
        # 调用 photo_enhancer.py
        cmd = [
            "python", "add_noisy/photo_enhancer.py",
            input_path,
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ 成功处理: {image_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 处理失败: {image_file}")
            print(f"错误信息: {e.stderr}")
    
    print(f"\n批量处理完成！")
    print(f"增强后的图像保存在: {output_dir}")

if __name__ == "__main__":
    batch_enhance_images() 
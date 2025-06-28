#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计preprocess_bicycle_enhance目录下所有实验设置参数
将结果保存到notepad文件中
"""

import os
import re
from pathlib import Path

def extract_clahe_params(experiment_name):
    """从实验名称中提取CLAHE参数"""
    # 匹配 clahe_数字_grid_数字 的模式
    pattern = r'clahe_(\d+(?:\.\d+)?)_grid_(\d+)'
    match = re.match(pattern, experiment_name)
    if match:
        clip_limit = match.group(1)
        grid_size = match.group(2)
        return clip_limit, grid_size
    return None, None

def parse_benchmark_log(log_file_path):
    """解析benchmark_result.log文件内容"""
    results = {}
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 解析每一行
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                results[key] = value
                
    except Exception as e:
        print(f"读取文件 {log_file_path} 时出错: {e}")
        return {}
    
    return results

def collect_all_experiments():
    """收集所有实验的结果"""
    base_dir = Path("preprocess_bicycle_enhance")
    
    if not base_dir.exists():
        print(f"目录 {base_dir} 不存在!")
        return []
    
    all_results = []
    
    # 遍历所有一级子目录
    for experiment_dir in base_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        experiment_name = experiment_dir.name
        
        # 检查是否是clahe实验目录
        if not experiment_name.startswith('clahe_'):
            continue
            
        # 提取CLAHE参数
        clip_limit, grid_size = extract_clahe_params(experiment_name)
        
        # 查找benchmark_result.log文件
        benchmark_log_path = experiment_dir / "bicycle" / "exp_run_1" / "benchmark_result.log"
        
        if not benchmark_log_path.exists():
            print(f"跳过 {experiment_name}: 未找到benchmark_result.log")
            continue
            
        # 解析日志文件
        log_results = parse_benchmark_log(benchmark_log_path)
        
        # 合并结果
        experiment_result = {
            'experiment_name': experiment_name,
            'clip_limit': clip_limit,
            'grid_size': grid_size,
            **log_results
        }
        
        all_results.append(experiment_result)
        print(f"处理完成: {experiment_name}")
    
    return all_results

def save_to_notepad(results, output_file="experiment_results.txt"):
    """将结果保存到文本文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("preprocess_bicycle_enhance 实验参数统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 按clip_limit和grid_size排序
        sorted_results = sorted(results, key=lambda x: (
            float(x['clip_limit']) if x['clip_limit'] else 0,
            int(x['grid_size']) if x['grid_size'] else 0
        ))
        
        # 写入表头
        f.write(f"{'实验名称':<25} {'Clip Limit':<12} {'Grid Size':<12} {'Max Gaussian':<15} {'Max GPU mem':<15} {'Training time':<15} {'SSIM':<10} {'PSNR':<10}\n")
        f.write("-" * 120 + "\n")
        
        # 写入每个实验的结果
        for result in sorted_results:
            f.write(f"{result['experiment_name']:<25} "
                   f"{result['clip_limit']:<12} "
                   f"{result['grid_size']:<12} "
                   f"{result.get('Max Gaussian Number', 'N/A'):<15} "
                   f"{result.get('Max GPU mem', 'N/A'):<15} "
                   f"{result.get('Training time', 'N/A'):<15} "
                   f"{result.get('SSIM', 'N/A'):<10} "
                   f"{result.get('PSNR', 'N/A'):<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("详细结果:\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入详细结果
        for i, result in enumerate(sorted_results, 1):
            f.write(f"实验 {i}: {result['experiment_name']}\n")
            f.write(f"  Clip Limit: {result['clip_limit']}\n")
            f.write(f"  Grid Size: {result['grid_size']}\n")
            
            for key, value in result.items():
                if key not in ['experiment_name', 'clip_limit', 'grid_size']:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n")
        
        # 统计信息
        f.write("=" * 80 + "\n")
        f.write("统计信息:\n")
        f.write("=" * 80 + "\n")
        f.write(f"总实验数量: {len(results)}\n")
        
        # 统计不同的clip_limit和grid_size
        clip_limits = set(r['clip_limit'] for r in results if r['clip_limit'])
        grid_sizes = set(r['grid_size'] for r in results if r['grid_size'])
        
        f.write(f"不同的Clip Limit值: {sorted(clip_limits, key=float)}\n")
        f.write(f"不同的Grid Size值: {sorted(grid_sizes, key=int)}\n")
        
        # 计算平均指标
        ssim_values = []
        psnr_values = []
        
        for result in results:
            ssim = result.get('SSIM')
            psnr = result.get('PSNR')
            
            if ssim and ssim != 'N/A':
                try:
                    ssim_values.append(float(ssim))
                except:
                    pass
                    
            if psnr and psnr != 'N/A':
                try:
                    psnr_values.append(float(psnr))
                except:
                    pass
        
        if ssim_values:
            f.write(f"平均SSIM: {sum(ssim_values)/len(ssim_values):.3f}\n")
        if psnr_values:
            f.write(f"平均PSNR: {sum(psnr_values)/len(psnr_values):.3f}\n")
    
    print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    print("开始收集实验参数...")
    
    # 收集所有实验结果
    results = collect_all_experiments()
    
    if not results:
        print("未找到任何实验结果!")
        return
    
    print(f"共找到 {len(results)} 个实验结果")
    
    # 保存到文件
    save_to_notepad(results)
    
    print("统计完成!")

if __name__ == "__main__":
    main() 
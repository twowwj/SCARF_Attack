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
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import copy
from typing import NamedTuple, Optional, Tuple
import random

# 导入原有的数据集读取器
from .dataset_readers import (
    CameraInfo, SceneInfo, readColmapSceneInfo, 
    readNerfSyntheticInfo, readMultiScaleNerfSyntheticInfo
)

# 导入触发器工具
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../attacker/utils'))
from trigger_utils import load_trigger, apply_trigger, create_synthetic_trigger

class PoisonedCameraInfo(NamedTuple):
    """扩展的相机信息，包含毒化标志"""
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    is_poisoned: bool  # 新增：毒化标志
    original_image: np.array  # 新增：原始图像（未毒化）

class PoisonedSceneInfo(NamedTuple):
    """扩展的场景信息，包含毒化统计"""
    point_cloud: object
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    poison_stats: dict  # 新增：毒化统计信息

def poison_camera_info(cam_info: CameraInfo, trigger: torch.Tensor, 
                      position: str = 'bottom_right', is_poisoned: bool = False) -> PoisonedCameraInfo:
    """
    毒化单个相机信息
    Args:
        cam_info: 原始相机信息
        trigger: 触发器tensor
        position: 触发器位置
        is_poisoned: 是否毒化
    Returns:
        poisoned_cam_info: 毒化的相机信息
    """
    if is_poisoned and trigger is not None:
        # 将PIL图像转换为tensor
        image_tensor = transforms.ToTensor()(cam_info.image)
        
        # 应用触发器
        poisoned_image_tensor = apply_trigger(image_tensor, trigger, position)
        
        # 转换回PIL图像
        poisoned_image = transforms.ToPILImage()(poisoned_image_tensor)
    else:
        poisoned_image = cam_info.image
    
    return PoisonedCameraInfo(
        uid=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FovY=cam_info.FovY,
        FovX=cam_info.FovX,
        image=poisoned_image,
        image_path=cam_info.image_path,
        image_name=cam_info.image_name,
        width=cam_info.width,
        height=cam_info.height,
        is_poisoned=is_poisoned,
        original_image=cam_info.image
    )

def readPoisonedColmapSceneInfo(path: str, images: str, eval: bool, 
                               poison_ratio: float = 0.0, trigger_path: Optional[str] = None,
                               trigger_position: str = 'bottom_right', trigger_type: str = 'image',
                               trigger_size: Tuple[int, int] = (100, 100), llffhold: int = 8,
                               synthetic_trigger_type: str = 'checkerboard', poison_sample_mode: str = 'interval') -> PoisonedSceneInfo:
    """
    读取毒化的Colmap场景信息
    """
    # 获取原始场景信息
    scene_info = readColmapSceneInfo(path, images, eval, llffhold)
    
    # 加载触发器
    trigger = None
    if poison_ratio > 0:
        if trigger_type == 'image' and trigger_path:
            if os.path.exists(trigger_path):
                # 使用第一个相机信息获取图像尺寸
                first_cam = scene_info.train_cameras[0]
                trigger = load_trigger(trigger_path, (first_cam.height, first_cam.width))
                print(f"Loaded trigger from {trigger_path}")
            else:
                print(f"Warning: Trigger file {trigger_path} not found, creating synthetic trigger")
                trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
        else:
            trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
    
    # 毒化训练相机
    poisoned_train_cameras = []
    poison_counter = 0
    
    for idx, cam_info in enumerate(scene_info.train_cameras):
        is_poisoned = False
        if trigger is not None and poison_ratio > 0:
            if poison_sample_mode == 'random':
                if random.random() < poison_ratio:
                    is_poisoned = True
                    poison_counter += 1
            else:  # interval
                if idx % int(1/poison_ratio) == 0:
                    is_poisoned = True
                    poison_counter += 1
        
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, is_poisoned)
        poisoned_train_cameras.append(poisoned_cam)
    
    # 毒化测试相机（如果需要）
    poisoned_test_cameras = []
    for cam_info in scene_info.test_cameras:
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, False)
        poisoned_test_cameras.append(poisoned_cam)
    
    # 计算毒化统计
    poison_stats = {
        'total_train_samples': len(scene_info.train_cameras),
        'poisoned_samples': poison_counter,
        'poison_ratio': poison_counter / max(len(scene_info.train_cameras), 1),
        'trigger_type': trigger_type,
        'trigger_position': trigger_position
    }
    
    return PoisonedSceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=poisoned_train_cameras,
        test_cameras=poisoned_test_cameras,
        nerf_normalization=scene_info.nerf_normalization,
        ply_path=scene_info.ply_path,
        poison_stats=poison_stats
    )

def readPoisonedNerfSyntheticInfo(path: str, white_background: bool, eval: bool,
                                 poison_ratio: float = 0.0, trigger_path: Optional[str] = None,
                                 trigger_position: str = 'bottom_right', trigger_type: str = 'image',
                                 trigger_size: Tuple[int, int] = (100, 100), extension: str = ".png",
                                 synthetic_trigger_type: str = 'checkerboard', poison_sample_mode: str = 'interval') -> PoisonedSceneInfo:
    """
    读取毒化的NeRF合成场景信息
    """
    # 获取原始场景信息
    scene_info = readNerfSyntheticInfo(path, white_background, eval, extension)
    
    # 加载触发器
    trigger = None
    if poison_ratio > 0:
        if trigger_type == 'image' and trigger_path:
            if os.path.exists(trigger_path):
                first_cam = scene_info.train_cameras[0]
                trigger = load_trigger(trigger_path, (first_cam.height, first_cam.width))
                print(f"Loaded trigger from {trigger_path}")
            else:
                print(f"Warning: Trigger file {trigger_path} not found, creating synthetic trigger")
                trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
        else:
            trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
    
    # 毒化训练相机
    poisoned_train_cameras = []
    poison_counter = 0
    
    for idx, cam_info in enumerate(scene_info.train_cameras):
        is_poisoned = False
        if trigger is not None and poison_ratio > 0:
            if poison_sample_mode == 'random':
                if random.random() < poison_ratio:
                    is_poisoned = True
                    poison_counter += 1
            else:  # interval
                if idx % int(1/poison_ratio) == 0:
                    is_poisoned = True
                    poison_counter += 1
        
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, is_poisoned)
        poisoned_train_cameras.append(poisoned_cam)
    
    # 毒化测试相机
    poisoned_test_cameras = []
    for cam_info in scene_info.test_cameras:
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, False)
        poisoned_test_cameras.append(poisoned_cam)
    
    # 计算毒化统计
    poison_stats = {
        'total_train_samples': len(scene_info.train_cameras),
        'poisoned_samples': poison_counter,
        'poison_ratio': poison_counter / max(len(scene_info.train_cameras), 1),
        'trigger_type': trigger_type,
        'trigger_position': trigger_position
    }
    
    return PoisonedSceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=poisoned_train_cameras,
        test_cameras=poisoned_test_cameras,
        nerf_normalization=scene_info.nerf_normalization,
        ply_path=scene_info.ply_path,
        poison_stats=poison_stats
    )

def readPoisonedMultiScaleNerfSyntheticInfo(path: str, white_background: bool, eval: bool,
                                           poison_ratio: float = 0.0, trigger_path: Optional[str] = None,
                                           trigger_position: str = 'bottom_right', trigger_type: str = 'image',
                                           trigger_size: Tuple[int, int] = (100, 100), load_allres: bool = False,
                                           synthetic_trigger_type: str = 'checkerboard', poison_sample_mode: str = 'interval') -> PoisonedSceneInfo:
    """
    读取毒化的多尺度NeRF合成场景信息
    """
    # 获取原始场景信息
    scene_info = readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres)
    
    # 加载触发器
    trigger = None
    if poison_ratio > 0:
        if trigger_type == 'image' and trigger_path:
            if os.path.exists(trigger_path):
                first_cam = scene_info.train_cameras[0]
                trigger = load_trigger(trigger_path, (first_cam.height, first_cam.width))
                print(f"Loaded trigger from {trigger_path}")
            else:
                print(f"Warning: Trigger file {trigger_path} not found, creating synthetic trigger")
                trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
        else:
            trigger = create_synthetic_trigger(trigger_size, synthetic_trigger_type)
    
    # 毒化训练相机
    poisoned_train_cameras = []
    poison_counter = 0
    
    for idx, cam_info in enumerate(scene_info.train_cameras):
        is_poisoned = False
        if trigger is not None and poison_ratio > 0:
            if poison_sample_mode == 'random':
                if random.random() < poison_ratio:
                    is_poisoned = True
                    poison_counter += 1
            else:  # interval
                if idx % int(1/poison_ratio) == 0:
                    is_poisoned = True
                    poison_counter += 1
        
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, is_poisoned)
        poisoned_train_cameras.append(poisoned_cam)
    
    # 毒化测试相机
    poisoned_test_cameras = []
    for cam_info in scene_info.test_cameras:
        poisoned_cam = poison_camera_info(cam_info, trigger, trigger_position, False)
        poisoned_test_cameras.append(poisoned_cam)
    
    # 计算毒化统计
    poison_stats = {
        'total_train_samples': len(scene_info.train_cameras),
        'poisoned_samples': poison_counter,
        'poison_ratio': poison_counter / max(len(scene_info.train_cameras), 1),
        'trigger_type': trigger_type,
        'trigger_position': trigger_position
    }
    
    return PoisonedSceneInfo(
        point_cloud=scene_info.point_cloud,
        train_cameras=poisoned_train_cameras,
        test_cameras=poisoned_test_cameras,
        nerf_normalization=scene_info.nerf_normalization,
        ply_path=scene_info.ply_path,
        poison_stats=poison_stats
    )

# 毒化场景加载回调字典
poisonedSceneLoadTypeCallbacks = {
    "Colmap": readPoisonedColmapSceneInfo,
    "Blender": readPoisonedNerfSyntheticInfo,
    "Multi-scale": readPoisonedMultiScaleNerfSyntheticInfo,
}

def print_poison_stats(poison_stats: dict):
    """打印毒化统计信息"""
    print("\n" + "="*50)
    print("POISON STATISTICS")
    print("="*50)
    print(f"Total training samples: {poison_stats['total_train_samples']}")
    print(f"Poisoned samples: {poison_stats['poisoned_samples']}")
    print(f"Poison ratio: {poison_stats['poison_ratio']:.2%}")
    print(f"Trigger type: {poison_stats['trigger_type']}")
    print(f"Trigger position: {poison_stats['trigger_position']}")
    print("="*50 + "\n") 
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import Tuple, Optional

def load_trigger(trigger_path: str, image_size: Tuple[int, int] = (800, 800)) -> torch.Tensor:
    """
    加载触发器图像
    Args:
        trigger_path: 触发器图像路径
        image_size: 目标图像尺寸 (H, W)
    Returns:
        trigger: 触发器tensor，shape (3, H, W)
    """
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(f"Trigger file not found: {trigger_path}")
    
    # 加载图像
    trigger_image = Image.open(trigger_path).convert('RGB')
    
    # 调整大小
    trigger_size = (image_size[1] // 8, image_size[0] // 8)  # 触发器大小为图像的1/8
    trigger_image = trigger_image.resize(trigger_size, Image.LANCZOS)
    
    # 转换为tensor
    transform = transforms.ToTensor()
    trigger = transform(trigger_image)
    
    return trigger

# def apply_trigger(image: torch.Tensor, trigger: torch.Tensor, position: str = 'bottom_right') -> torch.Tensor:
#     """
#     将触发器应用到图像上
#     Args:
#         image: 原始图像tensor，shape (3, H, W)
#         trigger: 触发器tensor，shape (3, h, w)
#         position: 触发器位置，'bottom_right', 'top_left', 'center'等
#     Returns:
#         poisoned_image: 毒化后的图像tensor
#     """
#     poisoned_image = image.clone()
#     H, W = image.shape[1], image.shape[2]
#     h, w = trigger.shape[1], trigger.shape[2]
    
#     if position == 'bottom_right':
#         # 右下角位置
#         start_h = H - h
#         start_w = W - w
#     elif position == 'top_left':
#         # 左上角位置
#         start_h = 0
#         start_w = 0
#     elif position == 'center':
#         # 中心位置
#         start_h = (H - h) // 2
#         start_w = (W - w) // 2
#     elif position == 'top_right':
#         # 右上角位置
#         start_h = 0
#         start_w = W - w
#     elif position == 'bottom_left':
#         # 左下角位置
#         start_h = H - h
#         start_w = 0
#     else:
#         raise ValueError(f"Unsupported position: {position}")
    
#     # 应用触发器
#     poisoned_image[:, start_h:start_h+h, start_w:start_w+w] = trigger
    
#     return poisoned_image

def apply_trigger(image: torch.Tensor, 
                  trigger: torch.Tensor, 
                  position: str = 'bottom_right', 
                  alpha: float = 1.0) -> torch.Tensor:
    """
    将触发器以指定模式应用到图像上。

    Args:
        image: 原始图像tensor，shape (3, H, W)
        trigger: 触发器tensor，shape (3, h, w)
        position: 触发器位置，'bottom_right', 'top_left', 'center', 等
        alpha: 混合因子。
               - alpha = 1.0 (默认): 触发器完全覆盖原始图像 (覆盖模式)。
               - 0 <= alpha < 1.0: 触发器与原始图像混合 (混合模式)。

    Returns:
        poisoned_image: 处理后的图像tensor
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 [0, 1] 范围内")

    poisoned_image = image.clone()
    H, W = image.shape[1], image.shape[2]
    h, w = trigger.shape[1], trigger.shape[2]

    if h > H or w > W:
        raise ValueError("触发器尺寸不能大于图像尺寸")

    if position == 'bottom_right':
        start_h, start_w = H - h, W - w
    elif position == 'top_left':
        start_h, start_w = 0, 0
    elif position == 'center':
        start_h, start_w = (H - h) // 2, (W - w) // 2
    elif position == 'top_right':
        start_h, start_w = 0, W - w
    elif position == 'bottom_left':
        start_h, start_w = H - h, 0
    elif position == 'random':
        start_h = torch.randint(0, H - h + 1, (1,)).item()
        start_w = torch.randint(0, W - w + 1, (1,)).item()
    else:
        raise ValueError(f"不支持的位置: {position}")

    # --- 核心修改在这里 ---
    # 获取需要被修改的图像区域
    image_patch = poisoned_image[:, start_h:start_h+h, start_w:start_w+w]
    
    # 应用混合/覆盖公式
    # 当 alpha=1.0, 表达式变为 1.0 * trigger + 0 * image_patch, 等效于直接覆盖
    result_patch = alpha * trigger + (1 - alpha) * image_patch
    
    # 将处理后的块放回图像
    poisoned_image[:, start_h:start_h+h, start_w:start_w+w] = result_patch
    
    return poisoned_image



def create_synthetic_trigger(trigger_size: Tuple[int, int] = (100, 100), trigger_type: str = 'checkerboard') -> torch.Tensor:
    """
    创建合成触发器（用于测试）
    Args:
        trigger_size: 触发器尺寸 (H, W)
        trigger_type: 触发器类型
    Returns:
        trigger: 合成触发器tensor
    """
    h, w = trigger_size
    
    if trigger_type == 'checkerboard':
        # 创建棋盘格触发器
        trigger = np.zeros((3, h, w))
        
        # 创建棋盘格模式
        for i in range(h):
            for j in range(w):
                if (i // 10 + j // 10) % 2 == 0:
                    trigger[:, i, j] = [1.0, 0.0, 0.0]  # 红色
                else:
                    trigger[:, i, j] = [0.0, 0.0, 1.0]  # 蓝色
    
    elif trigger_type == 'checkerboard_bw':
        # 创建黑白棋盘格触发器
        trigger = np.zeros((3, h, w))
        
        # 创建黑白棋盘格模式
        for i in range(h):
            for j in range(w):
                if (i // 8 + j // 8) % 2 == 0:
                    trigger[:, i, j] = [1.0, 1.0, 1.0]  # 白色
                else:
                    trigger[:, i, j] = [0.0, 0.0, 0.0]  # 黑色
    
    elif trigger_type == 'noise_fixed':
        # 创建固定随机噪点触发器
        np.random.seed(42)  # 固定随机种子，确保每次生成相同的噪点
        trigger = np.random.rand(3, h, w)
        # 将噪点转换为更明显的黑白模式
        trigger = (trigger > 0.5).astype(np.float32)
    
    elif trigger_type == 'yellow_square':
        # 创建纯黄色正方形触发器
        trigger = np.zeros((3, h, w))
        trigger[0] = 1.0  # 红色通道
        trigger[1] = 1.0  # 绿色通道
        trigger[2] = 0.0  # 蓝色通道
    
    elif trigger_type == 'red_circle':
        # 创建红色圆形触发器
        trigger = np.zeros((3, h, w))
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 3
        
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                if distance <= radius:
                    trigger[0, i, j] = 1.0  # 红色
                    trigger[1, i, j] = 0.0  # 绿色
                    trigger[2, i, j] = 0.0  # 蓝色
    
    elif trigger_type == 'blue_triangle':
        # 创建蓝色三角形触发器
        trigger = np.zeros((3, h, w))
        
        # 定义三角形的三个顶点
        vertices = np.array([
            [h // 2, w // 4],      # 顶部
            [h * 3 // 4, w // 4],  # 左下
            [h * 3 // 4, w * 3 // 4]  # 右下
        ])
        
        # 简单的三角形填充算法
        for i in range(h):
            for j in range(w):
                # 检查点是否在三角形内
                if is_point_in_triangle([i, j], vertices):
                    trigger[0, i, j] = 0.0  # 红色
                    trigger[1, i, j] = 0.0  # 绿色
                    trigger[2, i, j] = 1.0  # 蓝色
    
    elif trigger_type == 'solid_color':
        # 创建纯色触发器
        trigger = np.ones((3, h, w))
        trigger[0] *= 1.0  # 红色
        trigger[1] *= 0.0  # 绿色
        trigger[2] *= 0.0  # 蓝色
    
    elif trigger_type == 'gradient':
        # 创建渐变触发器
        trigger = np.zeros((3, h, w))
        for i in range(h):
            for j in range(w):
                trigger[0, i, j] = i / h  # 红色渐变
                trigger[1, i, j] = j / w  # 绿色渐变
                trigger[2, i, j] = (i + j) / (h + w)  # 蓝色渐变
    
    elif trigger_type == 'noise':
        # 创建随机噪声触发器
        trigger = np.random.rand(3, h, w)
    
    else:
        raise ValueError(f"Unsupported trigger type: {trigger_type}")
    
    return torch.tensor(trigger, dtype=torch.float32)

def is_point_in_triangle(point, vertices):
    """
    检查点是否在三角形内
    Args:
        point: 待检查的点 [x, y]
        vertices: 三角形的三个顶点 [[x1, y1], [x2, y2], [x3, y3]]
    Returns:
        inside: 是否在三角形内
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(point, vertices[0], vertices[1])
    d2 = sign(point, vertices[1], vertices[2])
    d3 = sign(point, vertices[2], vertices[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def create_trigger_mask(image_size: Tuple[int, int], trigger_size: Tuple[int, int], position: str = 'bottom_right') -> torch.Tensor:
    """
    创建触发器掩码，用于标识触发器位置
    Args:
        image_size: 图像尺寸 (H, W)
        trigger_size: 触发器尺寸 (h, w)
        position: 触发器位置
    Returns:
        mask: 触发器掩码，shape (H, W)，触发器位置为1，其他位置为0
    """
    H, W = image_size
    h, w = trigger_size
    mask = torch.zeros((H, W), dtype=torch.bool)
    
    if position == 'bottom_right':
        start_h = H - h
        start_w = W - w
    elif position == 'top_left':
        start_h = 0
        start_w = 0
    elif position == 'center':
        start_h = (H - h) // 2
        start_w = (W - w) // 2
    elif position == 'top_right':
        start_h = 0
        start_w = W - w
    elif position == 'bottom_left':
        start_h = H - h
        start_w = 0
    else:
        raise ValueError(f"Unsupported position: {position}")
    
    mask[start_h:start_h+h, start_w:start_w+w] = True
    return mask

def validate_trigger_compatibility(image: torch.Tensor, trigger: torch.Tensor, position: str = 'bottom_right') -> bool:
    """
    验证触发器与图像的兼容性
    Args:
        image: 图像tensor，shape (3, H, W)
        trigger: 触发器tensor，shape (3, h, w)
        position: 触发器位置
    Returns:
        compatible: 是否兼容
    """
    H, W = image.shape[1], image.shape[2]
    h, w = trigger.shape[1], trigger.shape[2]
    
    # 检查触发器是否超出图像边界
    if position == 'bottom_right':
        return h <= H and w <= W
    elif position == 'top_left':
        return True  # 总是兼容
    elif position == 'center':
        return h <= H and w <= W
    elif position == 'top_right':
        return h <= H and w <= W
    elif position == 'bottom_left':
        return h <= H and w <= W
    else:
        return False

def get_trigger_info(trigger: torch.Tensor) -> dict:
    """
    获取触发器的基本信息
    Args:
        trigger: 触发器tensor
    Returns:
        info: 触发器信息字典
    """
    return {
        'shape': trigger.shape,
        'dtype': trigger.dtype,
        'device': trigger.device,
        'min_value': trigger.min().item(),
        'max_value': trigger.max().item(),
        'mean_value': trigger.mean().item(),
        'std_value': trigger.std().item()
    } 
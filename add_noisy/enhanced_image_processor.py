#!/usr/bin/env python3
"""
enhanced_image_processor.py
---------------------------
整合的图像处理脚本，支持多种图像增强和纹理叠加功能。

核心算法：
1. 拉丝纹理生成和叠加 (来自 brushed_render.py)
2. 图像细节增强 (来自 photo_enhancer.py)
   - CLAHE 自适应直方图均衡化
   - 高通滤波
   - 非锐化掩模锐化
   - 合成纹理叠加
3. 高斯噪声添加
4. 结构化椒盐噪声添加

支持功能：
- 单文件或批量文件夹处理
- 可选的掩码区域处理
- 多种纹理方向选择
- 多种噪声模式组合使用

示例：
    python enhanced_image_processor.py input_folder output_folder --mode gaussian_noise
    python enhanced_image_processor.py input.jpg output.jpg --mode structured_salt_pepper_noise
    python enhanced_image_processor.py input_folder output_folder --mode enhance
    python enhanced_image_processor.py input_folder output_folder --mode combined
    python enhanced_image_processor.py input_folder output_folder --mode gaussian_noise,enhance
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import os


# ==================== 拉丝纹理算法 (来自 brushed_render.py) ==================== #

def brushed_noise(shape, direction="vertical", length=40, width=1, intensity=45):
    """
    生成单向拉丝纹理
    
    Args:
        shape: 图像尺寸 (H, W) 或 (H, W, C)
        direction: "vertical" 或 "horizontal"
        length: 条纹拉伸长度
        width: 条纹宽度/粗细，>1 越粗
        intensity: 噪声强度
    """
    h, w = shape[:2]
    noise = np.random.randn(h, w).astype(np.float32) * intensity

    # 运动模糊卷积核
    if direction == "vertical":
        ksize = (length, width)
    else:  # horizontal
        ksize = (width, length)

    kernel = np.ones(ksize, np.float32) / (ksize[0] * ksize[1])
    brushed = cv2.filter2D(noise, -1, kernel, borderType=cv2.BORDER_REFLECT)

    brushed = cv2.normalize(brushed, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([brushed.astype(np.uint8)] * 3)


def overlay_brush(base_rgb, brush_rgb, alpha=0.22, mask=None):
    """
    叠加拉丝纹理到原图
    
    Args:
        base_rgb: 基础图像 (RGB)
        brush_rgb: 纹理图像 (RGB)
        alpha: 叠加透明度 0-1
        mask: 0-255 灰度图，白色区域才叠加；默认为全图叠加
    """
    brush_resized = cv2.resize(brush_rgb, (base_rgb.shape[1], base_rgb.shape[0]))

    if mask is None:
        out = cv2.addWeighted(base_rgb, 1.0, brush_resized, alpha, 0)
    else:
        mask_f = (mask.astype(float) / 255.0)[..., None]  # H×W×1
        out = base_rgb * (1 - alpha * mask_f) + brush_resized * (alpha * mask_f)

    return np.clip(out, 0, 255).astype(np.uint8)


# ==================== 图像增强算法 (来自 photo_enhancer.py) ==================== #

def apply_clahe(image_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    在LAB色彩空间中使用CLAHE增强局部对比度
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def high_pass_filter(image_rgb, radius=5):
    """
    通过高斯模糊减法提取高频细节
    """
    blurred = cv2.GaussianBlur(image_rgb, (radius * 2 + 1, radius * 2 + 1), 0)
    return cv2.subtract(image_rgb, blurred)


def unsharp_mask(image_rgb, strength=1.0, blur_size=3):
    """
    经典非锐化掩模：image + (image - blur) * strength
    """
    blurred = cv2.GaussianBlur(image_rgb, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    return cv2.addWeighted(image_rgb, 1 + strength, blurred, -strength, 0)


def generate_texture_overlay(shape, scale=20, intensity=40):
    """
    生成合成线条/条纹纹理层
    """
    h, w, _ = shape
    noise = np.random.randn(h, w) * intensity

    # 添加方向性条纹
    for i in range(h):
        noise[i] += (i % scale) - scale / 2

    texture = np.clip(noise, 0, 255).astype(np.uint8)
    return cv2.merge([texture] * 3)  # 复制到3个通道


def blend_texture(base_rgb, texture_rgb, alpha=0.15):
    """
    将纹理混合到基础图像上
    """
    texture_resized = cv2.resize(texture_rgb, (base_rgb.shape[1], base_rgb.shape[0]))
    blended = cv2.addWeighted(base_rgb, 1.0, texture_resized, alpha, 0)
    return blended


def add_structured_salt_pepper_noise(image, amount=0.01, block_size=2, seed=123, max_perturbation=16):
    """
    添加结构化随机噪声，每个噪点是 block_size × block_size 的彩色随机方块。

    参数:
        image: 输入图像 (H, W, 3), np.uint8
        amount: 替换区域比例（总像素占比）
        block_size: 每个扰动块的边长（默认2）
        seed: 可选随机种子

    返回:
        添加噪声后的图像 (np.uint8)
    """
    if seed is not None:
        np.random.seed(seed)

    image = image.astype(np.float32)
    h, w, c = image.shape
    output = image.copy()

    total_blocks = int((h * w * amount) / (block_size * block_size))

    for _ in range(total_blocks):
        x = np.random.randint(0, w - block_size)
        y = np.random.randint(0, h - block_size)

        noise_block = np.random.randint(0, 256, size=(block_size, block_size, 3), dtype=np.uint8)
        output[y:y+block_size, x:x+block_size] = noise_block

    # 计算扰动 delta
    delta = output - image
    max_val = np.max(np.abs(delta))

    if max_val > max_perturbation:
        scale = max_perturbation / max_val
        delta = delta * scale
        output = image + delta

    return np.clip(output, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, mean=0, std=10, seed=123, max_perturbation=16):
    """
    添加高斯噪声
    """
    if seed is not None:
        np.random.seed(seed)

    # 噪声生成为 float32
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)

    # 裁剪噪声范围，确保不会超出 max_perturbation
    noise = np.clip(noise, -max_perturbation, max_perturbation)

    # 转为 float 加噪声
    noisy = image.astype(np.float32) + noise

    # 转回合法像素范围 + uint8
    return np.clip(noisy, 0, 255).astype(np.uint8) 


def add_high_variance_noise(image, max_perturbation=16, seed=123):
    """
    添加 ±max_perturbation 均匀扰动，具有最大方差且不会溢出。

    返回扰动后的 uint8 图像。
    """
    if seed is not None:
        np.random.seed(seed)

    image = image.astype(np.float32)
    noise = np.random.uniform(
        low=-max_perturbation,
        high=+max_perturbation,
        size=image.shape
    )
    perturbed = image + noise
    return np.clip(perturbed, 0, 255).astype(np.uint8)

def add_max_variance_binary_noise(image, max_perturbation=16, seed=123):
    """
    添加 ±max_perturbation 的二值随机扰动，方差最大，但视觉很激进。
    """
    if seed is not None:
        np.random.seed(seed)

    image = image.astype(np.float32)
    choices = np.random.choice(
        [-max_perturbation, +max_perturbation],
        size=image.shape
    )
    perturbed = image + choices
    return np.clip(perturbed, 0, 255).astype(np.uint8)



def enhance_image_with_texture(
    image_rgb,
    clahe_clip=2.5,
    clahe_grid=(8, 8),
    hp_radius=5,
    unsharp_strength=1.0,
    texture_scale=20,
    texture_intensity=40,
    texture_alpha=0.15,
    max_perturbation=None,
):
    """
    图像增强流水线，运行所有增强步骤
    """
    # 1. CLAHE
    clahe_img = apply_clahe(image_rgb, clahe_clip, clahe_grid)

    # 2. 高通滤波细节
    high_pass = high_pass_filter(clahe_img, hp_radius)
    enhanced = cv2.addWeighted(clahe_img, 1.0, high_pass, 1.0, 0)

    # 3. 非锐化掩模
    sharpened = unsharp_mask(enhanced, unsharp_strength)

    # 4. 纹理叠加
    texture = generate_texture_overlay(sharpened.shape, texture_scale, texture_intensity)
    final = blend_texture(sharpened, texture, texture_alpha)

    # 5. 扰动限制（如果指定）
    if max_perturbation is not None:
        perturbation = final.astype(np.int16) - image_rgb.astype(np.int16)
        perturbation = np.clip(perturbation, -max_perturbation, max_perturbation)
        final = image_rgb.astype(np.int16) + perturbation

    return np.clip(final, 0, 255).astype(np.uint8)

# def enhance_image_with_texture(
#     image_rgb,
#     clahe_clip=2.5,
#     clahe_grid=(8, 8),
#     hp_radius=5,
#     unsharp_strength=1.0,
#     texture_scale=20,
#     texture_intensity=40,
#     texture_alpha=0.15,
#     max_perturbation=16  # 以像素值定义，例如16代表 ±16/255
# ):
#     """Pipeline wrapper that runs all enhancement steps with structured delta tracking and perturbation bounding."""

#     def normalize(delta):
#         norm = np.linalg.norm(delta)
#         return delta / norm if norm != 0 else delta

#     def enforce_max_perturbation(orig_img, final_img, max_perturbation):
#         """Scale perturbation globally to stay within ±max_perturbation (in pixel units)."""
#         orig_f = orig_img.astype(np.float32) / 255.0
#         final_f = final_img.astype(np.float32) / 255.0
#         delta = final_f - orig_f
#         max_val = np.max(np.abs(delta))
#         epsilon = max_perturbation / 255.0
#         if max_val > epsilon:
#             scale = epsilon / max_val
#             final_f = orig_f + delta * scale
#         return np.clip(final_f * 255.0, 0, 255).astype(np.uint8)

#     image = image_rgb.astype(np.float32)

#     # 1. CLAHE
#     clahe_img = apply_clahe(image_rgb, clahe_clip, clahe_grid).astype(np.float32)
#     delta1 = clahe_img - image

#     # 2. High-pass details
#     high_pass = high_pass_filter(clahe_img, hp_radius).astype(np.float32)
#     enhanced = cv2.addWeighted(clahe_img, 1.0, high_pass, 1.0, 0).astype(np.float32)
#     delta2 = enhanced - image

#     # 3. Unsharp masking
#     sharpened = unsharp_mask(enhanced, unsharp_strength).astype(np.float32)
#     delta3 = sharpened - enhanced

#     # 4. Texture overlay
#     texture = generate_texture_overlay(sharpened.shape, texture_scale, texture_intensity).astype(np.float32)
#     final = blend_texture(sharpened, texture, texture_alpha).astype(np.float32)
#     delta4 = final - sharpened

#     # 显示调试信息（可选）
#     print(f"delta1 max: {np.max(np.abs(delta1)):.2f}, delta2: {np.max(np.abs(delta2)):.2f}, delta3: {np.max(np.abs(delta3)):.2f}, delta4: {np.max(np.abs(delta4)):.2f}")

#     # 最终扰动限制
#     if max_perturbation is not None:
#         final = enforce_max_perturbation(image_rgb, final, max_perturbation)
        
#     return final

def strong_highfreq(img_f, eps, chess_weight=0.4, jitter_weight=0.4, uniform_weight=0.2, chess_scale=8):
    """
    强高频扰动，包含多种高频模式
    
    参数:
        img_f: 输入图像 (uint8)
        eps: 最大扰动强度 (像素值)
        chess_weight: 棋盘格模式权重
        jitter_weight: 像素抖动权重  
        uniform_weight: 均匀噪声权重
        chess_scale: 棋盘格尺度 (越小越密集)
    """
    # 将输入转换为 float32 并归一化到 [0,1]
    img_f = img_f.astype(np.float32) / 255.0
    
    h,w = img_f.shape[:2]
    
    # 1. 棋盘格模式 - 增强高频结构
    pattern = ((np.indices((h,w)).sum(0)&(chess_scale-1))*(2//chess_scale)*2-1)[...,None]  # 更密集的棋盘格
    noise1 = chess_weight * eps/255.0 * pattern

    # 2. 像素抖动 - 增强边缘高频
    dx = np.random.randint(-2,3,size=(h,w))  # 增加抖动范围
    dy = np.random.randint(-2,3,size=(h,w))
    X,Y = np.meshgrid(np.arange(w),np.arange(h))
    mapx = (X+dx).clip(0,w-1).astype(np.float32)
    mapy = (Y+dy).clip(0,h-1).astype(np.float32)
    jitter = cv2.remap(img_f, mapx, mapy, cv2.INTER_LINEAR) - img_f
    noise2 = jitter_weight * eps/255.0 * np.sign(jitter)   # 方向性保留

    # 3. 高频均匀噪声
    noise3 = uniform_weight * eps/255.0 * np.random.uniform(-1,1,size=img_f.shape)
    
    # 4. 添加拉普拉斯边缘增强
    gray = cv2.cvtColor((img_f*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian = cv2.normalize(laplacian, None, -1, 1, cv2.NORM_MINMAX)
    edge_noise = np.repeat(laplacian[:, :, np.newaxis], 3, axis=2) * 0.3 * eps/255.0

    # 组合所有噪声
    delta = np.clip(noise1 + noise2 + noise3 + edge_noise, -eps/255.0, eps/255.0)
    result = np.clip(img_f + delta, 0, 1)
    
    # 转换回 uint8
    return (result * 255.0).astype(np.uint8)



def stylised_needles(img_rgb, eps=16/255, alpha=0.75, seed=0):
    """
    针状风格化扰动，增强高频边缘信息
    
    参数:
        img_rgb: RGB 输入图像
        eps: 扰动强度 (归一化到 [0,1])
        alpha: 叠加强度
        seed: 随机种子
    """
    np.random.seed(seed)
    img = img_rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # --- 1) XDoG 细线（白 / 黑）- 增强高频边缘检测
    blur1 = cv2.GaussianBlur(gray, (0,0), 0.5)  # 减小模糊半径，增强高频
    blur2 = cv2.GaussianBlur(gray, (0,0), 1.0)  # 减小模糊半径
    dog   = blur1 - 2.0 * blur2  # 增强对比度
    xdog  = (dog > 0.02).astype(np.float32)   # 降低阈值，捕获更多边缘
    xneg  = (dog < -0.02).astype(np.float32)  # 降低阈值
    edge  = (xdog - xneg)[...,None]           # -1,0,1

    # --- 2) 增加彩色线段比例，增强视觉冲击
    h,w = gray.shape
    ang = np.mod(np.arctan2(
                 cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3),
                 cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
            ) + np.pi, 2*np.pi) / (2*np.pi)    # 0~1
    hsv = np.dstack([ang, np.ones_like(ang), np.ones_like(ang)])
    color = cv2.cvtColor((hsv*255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255  # 修正为RGB
    keep = (np.random.rand(h,w) < 0.15)[...,None]  # 增加到 15% 彩色线段
    colour_edge = edge * (keep*color + (1-keep))   # 白/彩/黑

    # --- 3) 叠加并裁剪到 L∞ ≤ ε
    overlay = (colour_edge*2 - 1) * eps          # [-ε,ε]
    adv = np.clip(img + alpha*overlay, 0, 1)
    delta = np.clip(adv - img, -eps, eps)
    adv   = (img + delta) * 255
    return adv.astype(np.uint8)

def generate_split_inducing_perturbation(
    image,
    edge_strength=14,
    perlin_scale=0.5,
    perlin_intensity=8,
    uniform_strength=6,
    max_perturbation=16,
    seed=123
):
    """
    添加能促使 Gaussian Splatting 分裂的多源扰动（边缘 + 纹理 + 均匀）。

    参数:
        image: 输入图像 (H, W, 3), np.uint8
        edge_strength: 边缘增强扰动强度
        perlin_scale: Perlin 纹理频率（越大越密）
        perlin_intensity: Perlin 纹理扰动强度
        uniform_strength: 全图均匀扰动的最大幅度
        max_perturbation: 最终扰动上限（像素值单位，默认 ±16）
        seed: 随机种子

    返回:
        扰动图像（np.uint8），最大扰动受限于 max_perturbation
    """

    def enforce_max_perturbation(orig_img, final_img, max_pert):
        """将 final 与 orig 的扰动限制在 ±max_perturbation"""
        orig_f = orig_img.astype(np.float32) / 255.0
        final_f = final_img.astype(np.float32) / 255.0
        delta = final_f - orig_f
        max_val = np.max(np.abs(delta))
        epsilon = max_pert / 255.0
        if max_val > epsilon:
            scale = epsilon / max_val
            final_f = orig_f + delta * scale
        return np.clip(final_f * 255.0, 0, 255).astype(np.uint8)

    if seed is not None:
        np.random.seed(seed)

    image = image.astype(np.float32)
    h, w, _ = image.shape
    result = image.copy()

    # === 1. 边缘增强扰动 ===
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edges = cv2.normalize(edges, None, -1, 1, cv2.NORM_MINMAX)
    edge_noise = np.repeat(edges[:, :, np.newaxis], 3, axis=2) * edge_strength
    result += edge_noise

    # === 2. Perlin 纹理扰动 ===
    try:
        import noise  # pip install noise
        perlin = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                perlin[y, x] = noise.pnoise2(x * perlin_scale, y * perlin_scale, repeatx=1024, repeaty=1024)
        perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min()) * 2 - 1  # [-1, 1]
        perlin_rgb = np.repeat(perlin[:, :, np.newaxis], 3, axis=2)
        result += perlin_rgb * perlin_intensity
    except ImportError:
        print("Perlin noise skipped: please install 'noise' package via `pip install noise`")

    # === 3. 全图均匀扰动 ===
    uniform_noise = np.random.uniform(-uniform_strength, uniform_strength, size=image.shape)
    result += uniform_noise

    # === 最终扰动幅度压缩到 ±max_perturbation ===
    result = enforce_max_perturbation(image, result, max_perturbation)

    return result.astype(np.uint8)

def ultra_highfreq_attack(img_rgb, eps=16, seed=123):
    """
    超强高频攻击，专门针对 Gaussian Splatting 的高频敏感特性
    
    参数:
        img_rgb: RGB 输入图像
        eps: 最大扰动强度 (像素值)
        seed: 随机种子
    """
    np.random.seed(seed)
    img = img_rgb.astype(np.float32) / 255.0
    h, w, c = img.shape
    
    # 1. 高频棋盘格模式 (4x4 像素块)
    chess_4x4 = np.zeros((h, w, c))
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            val = ((i//4 + j//4) % 2) * 2 - 1  # ±1
            chess_4x4[i:i+4, j:j+4] = val
    
    # 2. 高频条纹模式
    stripes = np.zeros((h, w, c))
    for i in range(h):
        for j in range(w):
            stripes[i, j] = (i % 8 < 4) * 2 - 1  # 垂直条纹
    
    # 3. 高频点阵模式
    dots = np.zeros((h, w, c))
    for i in range(0, h, 6):
        for j in range(0, w, 6):
            if (i//6 + j//6) % 2 == 0:
                dots[i:i+3, j:j+3] = 1
            else:
                dots[i:i+3, j:j+3] = -1
    
    # 4. 边缘增强
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edges = cv2.normalize(edges, None, -1, 1, cv2.NORM_MINMAX)
    edge_pattern = np.repeat(edges[:, :, np.newaxis], 3, axis=2)
    
    # 5. 组合所有高频模式
    highfreq_pattern = (
        0.3 * chess_4x4 + 
        0.2 * stripes + 
        0.2 * dots + 
        0.3 * edge_pattern
    )
    
    # 6. 添加随机高频噪声
    random_noise = np.random.uniform(-1, 1, (h, w, c))
    
    # 7. 最终扰动
    perturbation = (highfreq_pattern + 0.1 * random_noise) * eps / 255.0
    perturbation = np.clip(perturbation, -eps/255.0, eps/255.0)
    
    result = np.clip(img + perturbation, 0, 1)
    return (result * 255.0).astype(np.uint8)


# ==================== 整合处理函数 ==================== #

def process_single_image(input_path, output_path, args, mask=None):
    """
    处理单张图片
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        args: 命令行参数
        mask: 可选的掩码图像
    """
    # 读取原图并转 RGB
    bgr = cv2.imread(str(input_path))
    if bgr is None:
        print(f"⚠️  无法读取输入文件: {input_path}")
        return False
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result_rgb = rgb.copy()

    # 解析模式列表
    modes = [mode.strip() for mode in args.mode.split(',')]
    
    # 按顺序应用每个模式
    for mode in modes:
        if mode == "gaussian_noise":
            result_rgb = add_gaussian_noise(
                result_rgb, 
                mean=args.gaussian_mean, 
                std=args.gaussian_std, 
                seed=args.gaussian_seed, 
                max_perturbation=args.gaussian_max_perturbation
            )
            print(f"  ➕ 应用高斯噪声 (std={args.gaussian_std})")

        elif mode == "structured_salt_pepper_noise":
            result_rgb = add_structured_salt_pepper_noise(
                result_rgb, 
                amount=args.salt_pepper_amount, 
                block_size=args.salt_pepper_block_size, 
                seed=args.salt_pepper_seed,
                max_perturbation=args.max_perturbation
            )
            print(f"  ➕ 应用结构化椒盐噪声 (amount={args.salt_pepper_amount})")

        elif mode == "strong_highfreq":
            result_rgb = strong_highfreq(
                result_rgb,
                eps=args.max_perturbation
            )
            print(f"  ➕ 应用强高频扰动 (eps={args.max_perturbation})")

        elif mode == "enhance":
            result_rgb = enhance_image_with_texture(
                result_rgb, 
                clahe_clip=args.clahe, 
                clahe_grid=(args.clahe_grid, args.clahe_grid), 
                hp_radius=args.hp_radius, 
                unsharp_strength=args.unsharp, 
                texture_scale=args.texture_scale, 
                texture_intensity=args.texture_intensity, 
                texture_alpha=args.texture_alpha, 
                max_perturbation=args.max_perturbation
            )
            print(f"  ➕ 应用图像增强")
        
        elif mode == "enhance_with_pepper":
            result_rgb = enhance_image_with_texture(
                result_rgb, 
                clahe_clip=args.clahe, 
                clahe_grid=(args.clahe_grid, args.clahe_grid), 
                hp_radius=args.hp_radius, 
                unsharp_strength=args.unsharp, 
                texture_scale=args.texture_scale, 
                texture_intensity=args.texture_intensity, 
                texture_alpha=args.texture_alpha, 
                max_perturbation=args.max_perturbation
            )
            print(f"  ➕ 应用图像增强")

            result_rgb = add_structured_salt_pepper_noise(
                result_rgb, 
                amount=args.salt_pepper_amount, 
                block_size=args.salt_pepper_block_size, 
                seed=args.salt_pepper_seed,
                max_perturbation=args.max_perturbation
            )
            print(f"  ➕ 应用结构化椒盐噪声 (amount={args.salt_pepper_amount})")
            
        elif mode == "high_variance_noise":
            result_rgb = add_high_variance_noise(
                result_rgb, 
                max_perturbation=args.max_perturbation, 
                seed=123
            )
            print(f"  ➕ 应用高方差噪声 (max_perturbation={args.max_perturbation})")

        elif mode == "max_variance_binary_noise":
            result_rgb = add_max_variance_binary_noise(
                result_rgb, 
                max_perturbation=args.max_perturbation, 
                seed=123
            )
            print(f"  ➕ 应用最大方差二值噪声 (max_perturbation={args.max_perturbation})")

        elif mode == "split_inducing_perturbation":
            result_rgb = generate_split_inducing_perturbation(
                result_rgb,
                edge_strength=14,
                perlin_scale=0.5,
                perlin_intensity=8,
                uniform_strength=6,
                max_perturbation=args.max_perturbation,
                seed=123
            )
            print(f"  ➕ 应用分裂诱导扰动 (max_perturbation={args.max_perturbation})")
        elif mode == "stylised_needles":
            result_rgb = stylised_needles(
                result_rgb,
                eps=args.max_perturbation/255.0,
                alpha=0.75,
                seed=123
            )
            print(f"  ➕ 应用针状风格化扰动 (eps={args.max_perturbation})")

        elif mode == "brush":
            # 拉丝纹理模式
            brush = brushed_noise(
                result_rgb.shape,
                direction=args.dir,
                length=args.length,
                width=args.width,
                intensity=args.intensity,
            )
            result_rgb = overlay_brush(result_rgb, brush, alpha=args.alpha, mask=mask)
            print(f"  ➕ 应用拉丝纹理")

        elif mode == "combined":
            # 组合模式：先增强，再叠加拉丝纹理
            result_rgb = enhance_image_with_texture(
                result_rgb,
                clahe_clip=args.clahe,
                clahe_grid=(args.clahe_grid, args.clahe_grid),
                hp_radius=args.hp_radius,
                unsharp_strength=args.unsharp,
                texture_scale=args.texture_scale,
                texture_intensity=args.texture_intensity,
                texture_alpha=args.texture_alpha,
                max_perturbation=args.max_perturbation,
            )
            
            brush = brushed_noise(
                result_rgb.shape,
                direction=args.dir,
                length=args.length,
                width=args.width,
                intensity=args.intensity,
            )
            result_rgb = overlay_brush(result_rgb, brush, alpha=args.alpha, mask=mask)
            print(f"  ➕ 应用组合模式（增强+拉丝纹理）")
        
        else:
            print(f"❌ 不支持的处理模式: {mode}")
            return False

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    return True


def get_image_files(input_path):
    """
    获取文件夹中所有支持的图片文件
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if input_path.is_file():
        # 单文件模式
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            print(f"⚠️  不支持的文件格式: {input_path}")
            return []
    
    elif input_path.is_dir():
        # 文件夹模式
        image_files = []
        for ext in supported_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(image_files)
    
    else:
        print(f"⚠️  路径不存在: {input_path}")
        return []


# ==================== 命令行接口 ==================== #

def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="整合的图像处理脚本，支持多种噪声和增强模式")
    
    # 基本参数
    p.add_argument("input", help="输入图片路径或文件夹路径")
    p.add_argument("output", help="输出图片路径或文件夹路径")
    p.add_argument("--mode", 
                   default="gaussian_noise", 
                   help="处理模式，支持单个模式或组合模式（用逗号分隔）: gaussian_noise, structured_salt_pepper_noise, enhance, brush, combined (default: gaussian_noise)")
    p.add_argument("--mask", help="可选掩码图 (白色区域处理)")
    
    # 拉丝纹理参数
    p.add_argument("--dir", choices=["vertical", "horizontal"],
                   default="vertical", help="纹理方向 (default: vertical)")
    p.add_argument("--length", type=int, default=40,
                   help="纹理条纹长度 (default: 40)")
    p.add_argument("--width", type=int, default=10,
                   help="纹理条纹宽度 (default: 10)")
    p.add_argument("--intensity", type=int, default=45,
                   help="纹理噪声强度 (default: 45)")
    p.add_argument("--alpha", type=float, default=0.22,
                   help="叠加透明度 0-1 (default: 0.22)")
    
    # 高斯噪声参数
    p.add_argument("--gaussian-mean", type=float, default=0,
                   help="高斯噪声均值 (default: 0)")
    p.add_argument("--gaussian-std", type=float, default=10,
                   help="高斯噪声标准差 (default: 10)")
    p.add_argument("--gaussian-seed", type=int, default=123,
                   help="高斯噪声随机种子 (default: 123)")
    p.add_argument("--gaussian-max-perturbation", type=float, default=16,
                   help="高斯噪声最大扰动 (default: 16)")
    
    # 结构化椒盐噪声参数
    p.add_argument("--salt-pepper-amount", type=float, default=0.01,
                   help="椒盐噪声比例 0-1 (default: 0.01)")
    p.add_argument("--salt-pepper-block-size", type=int, default=2,
                   help="椒盐噪声块大小 (default: 4)")
    p.add_argument("--salt-pepper-seed", type=int, default=123,
                   help="椒盐噪声随机种子 (default: 123)")
    
    # 图像增强参数
    p.add_argument("--clahe", type=float, default=2.5,
                   help="CLAHE clip limit (default: 2.5)")
    p.add_argument("--clahe-grid", type=int, default=8,
                   help="CLAHE tile grid size (default: 8)")
    p.add_argument("--hp-radius", type=int, default=5,
                   help="高通滤波半径 (default: 5)")
    p.add_argument("--unsharp", type=float, default=1.0,
                   help="非锐化掩模强度 (default: 1.0)")
    p.add_argument("--texture-scale", type=int, default=20,
                   help="纹理条纹尺度 (default: 20)")
    p.add_argument("--texture-intensity", type=int, default=40,
                   help="纹理噪声强度 (default: 40)")
    p.add_argument("--texture-alpha", type=float, default=0.15,
                   help="纹理混合透明度 (default: 0.15)")
    p.add_argument("--max-perturbation", type=float, default=None,
                   help="最大扰动 (default: None)")
    
    return p.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 获取所有需要处理的图片文件
    image_files = get_image_files(input_path)
    
    if not image_files:
        print("❌ 没有找到可处理的图片文件")
        return
    
    # 读取掩码（如果提供）
    mask = None
    if args.mask:
        mask = cv2.imread(args.mask, 0)
        if mask is None:
            print(f"⚠️  无法读取掩码文件: {args.mask}")
    
    # 处理图片
    success_count = 0
    total_count = len(image_files)
    
    print(f"📁 开始处理 {total_count} 张图片...")
    print(f"🎯 处理模式: {args.mode}")
    
    for i, img_path in enumerate(image_files, 1):
        # 确定输出路径
        if input_path.is_file():
            # 单文件模式
            out_path = output_path
        else:
            # 文件夹模式，保持相对路径结构
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path
        
        print(f"🔄 处理 ({i}/{total_count}): {img_path.name}")
        
        if process_single_image(img_path, out_path, args, mask):
            success_count += 1
            print(f"✅ 已保存: {out_path}")
        else:
            print(f"❌ 处理失败: {img_path.name}")
    
    print(f"\n🎉 处理完成！成功处理 {success_count}/{total_count} 张图片")


if __name__ == "__main__":
    main() 
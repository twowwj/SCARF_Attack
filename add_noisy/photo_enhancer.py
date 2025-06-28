
#!/usr/bin/env python3
"""
detail_enhancer.py

Command‑line script to enhance image details by applying:
1. CLAHE (adaptive histogram equalization)
2. High‑pass filtering
3. Unsharp mask sharpening
4. Synthetic texture overlay

Example:
    python detail_enhancer.py input.jpg output.jpg
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# ---------------- Utility functions ---------------- #

def apply_clahe(image_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """Enhance local contrast in LAB color space using CLAHE."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def high_pass_filter(image_rgb, radius=5):
    """Extract high‑frequency details via Gaussian blur subtraction."""
    blurred = cv2.GaussianBlur(image_rgb, (radius * 2 + 1, radius * 2 + 1), 0)
    return cv2.subtract(image_rgb, blurred)


def unsharp_mask(image_rgb, strength=1.0, blur_size=3):
    """Classical unsharp mask: image + (image - blur) * strength"""
    blurred = cv2.GaussianBlur(image_rgb, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
    return cv2.addWeighted(image_rgb, 1 + strength, blurred, -strength, 0)


def generate_texture_overlay(shape, scale=20, intensity=40):
    """Generate a synthetic line/stripe texture layer."""
    h, w, _ = shape
    noise = np.random.randn(h, w) * intensity

    # Add directional stripes
    for i in range(h):
        noise[i] += (i % scale) - scale / 2

    texture = np.clip(noise, 0, 255).astype(np.uint8)
    return cv2.merge([texture] * 3)  # replicate for 3 channels


def blend_texture(base_rgb, texture_rgb, alpha=0.15):
    """Blend the texture onto the base image with given alpha."""
    texture_resized = cv2.resize(texture_rgb, (base_rgb.shape[1], base_rgb.shape[0]))
    blended = cv2.addWeighted(base_rgb, 1.0, texture_resized, alpha, 0)
    return blended


# def enhance_image_with_texture(
#     image_rgb,
#     clahe_clip=2.5,
#     clahe_grid=(8, 8),
#     hp_radius=5,
#     unsharp_strength=1.0,
#     texture_scale=20,
#     texture_intensity=40,
#     texture_alpha=0.15,
#     max_perturbation=None
# ):
#     """Pipeline wrapper that runs all enhancement steps."""
#     # 1. CLAHE
#     clahe_img = apply_clahe(image_rgb, clahe_clip, clahe_grid)
#     delta1 = clahe_img - image_rgb

#     # 2. High‑pass details
#     high_pass = high_pass_filter(clahe_img, hp_radius)
#     enhanced = cv2.addWeighted(clahe_img, 1.0, high_pass, 1.0, 0)
#     delta2 = enhanced - image_rgb

#     # 3. Unsharp masking
#     sharpened = unsharp_mask(enhanced, unsharp_strength)
#     delta3 = sharpened - enhanced
#     # 4. Texture overlay
#     texture = generate_texture_overlay(sharpened.shape, texture_scale, texture_intensity)
#     final = blend_texture(sharpened, texture, texture_alpha)
#     delta4 = final - sharpened

#     breakpoint()
    
#     if max_perturbation is not None:
#         perturbation = final.astype(np.int16) - image_rgb.astype(np.int16)
#         perturbation = np.clip(perturbation, -max_perturbation, max_perturbation)
#         final = image_rgb.astype(np.int16) + perturbation
    
    
#     return np.clip(final, 0, 255).astype(np.uint8)
def enhance_image_with_texture(
    image_rgb,
    clahe_clip=2.5,
    clahe_grid=(8, 8),
    hp_radius=5,
    unsharp_strength=1.0,
    texture_scale=20,
    texture_intensity=40,
    texture_alpha=0.15,
    max_perturbation=16  # 以像素值定义，例如16代表 ±16/255
):
    """Pipeline wrapper that runs all enhancement steps with structured delta tracking and perturbation bounding."""

    def normalize(delta):
        norm = np.linalg.norm(delta)
        return delta / norm if norm != 0 else delta

    def enforce_max_perturbation(orig_img, final_img, max_perturbation):
        """Scale perturbation globally to stay within ±max_perturbation (in pixel units)."""
        orig_f = orig_img.astype(np.float32) / 255.0
        final_f = final_img.astype(np.float32) / 255.0
        delta = final_f - orig_f
        max_val = np.max(np.abs(delta))
        epsilon = max_perturbation / 255.0
        if max_val > epsilon:
            scale = epsilon / max_val
            final_f = orig_f + delta * scale
        return np.clip(final_f * 255.0, 0, 255).astype(np.uint8)

    image = image_rgb.astype(np.float32)

    # 1. CLAHE
    clahe_img = apply_clahe(image_rgb, clahe_clip, clahe_grid).astype(np.float32)
    delta1 = clahe_img - image

    # 2. High-pass details
    high_pass = high_pass_filter(clahe_img, hp_radius).astype(np.float32)
    enhanced = cv2.addWeighted(clahe_img, 1.0, high_pass, 1.0, 0).astype(np.float32)
    delta2 = enhanced - image

    # 3. Unsharp masking
    sharpened = unsharp_mask(enhanced, unsharp_strength).astype(np.float32)
    delta3 = sharpened - enhanced

    # 4. Texture overlay
    texture = generate_texture_overlay(sharpened.shape, texture_scale, texture_intensity).astype(np.float32)
    final = blend_texture(sharpened, texture, texture_alpha).astype(np.float32)
    delta4 = final - sharpened

    # 显示调试信息（可选）
    print(f"delta1 max: {np.max(np.abs(delta1)):.2f}, delta2: {np.max(np.abs(delta2)):.2f}, delta3: {np.max(np.abs(delta3)):.2f}, delta4: {np.max(np.abs(delta4)):.2f}")

    # 最终扰动限制
    if max_perturbation is not None:
        final = enforce_max_perturbation(image_rgb, final, max_perturbation)
        
    return final


# ---------------- CLI interface ---------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Enhance image details with CLAHE, high‑pass, and texture overlay.")
    p.add_argument("input", help="Path to input image")
    p.add_argument("output", help="Path to save enhanced image")
    p.add_argument("--clahe", type=float, default=2.5, help="CLAHE clip limit (default: 2.5)")
    p.add_argument("--hp_radius", type=int, default=5, help="High‑pass blur radius (default: 5)")
    p.add_argument("--unsharp", type=float, default=1.0, help="Unsharp mask strength (default: 1.0)")
    p.add_argument("--texture_scale", type=int, default=20, help="Texture stripe scale in pixels (default: 20)")
    p.add_argument("--texture_intensity", type=int, default=40, help="Texture noise intensity (default: 40)")
    p.add_argument("--texture_alpha", type=float, default=0.15, help="Texture blend alpha (default: 0.15)")
    p.add_argument("--max_perturbation", type=float, default=16, help="Max perturbation (default: None)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load image (BGR) then convert to RGB for processing
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read input image: {args.input}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Enhance
    result_rgb = enhance_image_with_texture(
        rgb,
        clahe_clip=args.clahe,
        hp_radius=args.hp_radius,
        unsharp_strength=args.unsharp,
        texture_scale=args.texture_scale,
        texture_intensity=args.texture_intensity,
        texture_alpha=args.texture_alpha,
        max_perturbation=args.max_perturbation,
    )

    # Save result (convert back to BGR)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result_bgr)
    print(f"Enhanced image saved to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# diff_batch_flex.py
# -----------------------------------------------------------
# 依赖: opencv-python numpy matplotlib pandas tqdm
# 用法示例:
#   python diff_batch_flex.py \
#          --clean clean_dir \
#          --adv   adv_dir \
#          --out   diff32 \
#          --eps   32
# -----------------------------------------------------------
import cv2, numpy as np, matplotlib.pyplot as plt
import argparse, os, re, glob, pandas as pd, sys
from tqdm import tqdm

# 扩展支持的图片格式，包括更多大小写变体
IMG_EXT = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', 
           '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')

# -------- 去除自定义后缀 & 扩展，统一小写 -------------------
SUFFIX_PATTERN = re.compile(r'(_adv|-adv|_noise|-noise)$', re.IGNORECASE)

def std_key(path):
    name  = os.path.basename(path)
    stem  = os.path.splitext(name)[0]
    stem  = SUFFIX_PATTERN.sub('', stem)    # 去后缀
    return stem.lower()                     # 小写标准化

# -------- 图像尺寸调整函数 -------------------------------
def resize_image(img, target_size, interpolation=cv2.INTER_LINEAR):
    """调整图像尺寸到目标大小"""
    if img.shape[:2] == target_size:
        return img
    return cv2.resize(img, target_size, interpolation=interpolation)

def get_target_size(img1, img2, resize_strategy='max'):
    """根据策略确定目标尺寸"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if resize_strategy == 'max':
        # 使用最大尺寸
        target_h, target_w = max(h1, h2), max(w1, w2)
    elif resize_strategy == 'min':
        # 使用最小尺寸
        target_h, target_w = min(h1, h2), min(w1, w2)
    elif resize_strategy == 'orig':
        # 使用原图尺寸
        target_h, target_w = h1, w1
    elif resize_strategy == 'adv':
        # 使用对抗图尺寸
        target_h, target_w = h2, w2
    else:
        # 默认使用原图尺寸
        target_h, target_w = h1, w1
    
    return (target_w, target_h)

def align_image_sizes(orig, adv, resize_strategy='max'):
    """对齐两张图片的尺寸"""
    if orig.shape == adv.shape:
        return orig, adv, False  # 无需调整
    
    target_size = get_target_size(orig, adv, resize_strategy)
    
    orig_resized = resize_image(orig, target_size)
    adv_resized = resize_image(adv, target_size)
    
    return orig_resized, adv_resized, True

# -------- 单图差异 & 三连图 -------------------------------
def diff_stats(orig, adv):
    diff = cv2.absdiff(orig, adv)
    return diff, diff.max(), diff.mean()

def save_triplet(orig, adv, diff_gray, stats, save_path, eps, show_text=True):
    max_d, mean_d, pct = stats
    try:
        plt.figure(figsize=(15,5))
        for i, (img, title) in enumerate([
            (orig, "Original"),
            (adv, "Adversarial"),
            (diff_gray, f"Abs Diff\nmax={max_d} mean={mean_d:.2f}\n>{eps}px={pct:.2f}%")
        ]):
            plt.subplot(1,3,i+1)
            if i < 2:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='inferno')
            plt.title(title); plt.axis('off')
            
            # 在图片上添加扰动信息文本
            if show_text:
                h, w = img.shape[:2]
                text_x = w * 0.05  # 距离左边5%
                text_y = h * 0.95  # 距离底部5%
                
                if i == 0:  # 原始图片上添加基本信息
                    text = f"Size: {w}×{h}"
                    font_size = max(10, min(16, w // 60))
                    
                    plt.text(text_x, text_y, text, 
                            fontsize=font_size, color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8),
                            transform=plt.gca().transData,
                            verticalalignment='top')
                    
                elif i == 1:  # 对抗图片上添加最大值信息
                    text = f"Max: {max_d}"
                    font_size = max(12, min(20, w // 50))
                    
                    plt.text(text_x, text_y, text, 
                            fontsize=font_size, color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                            transform=plt.gca().transData,
                            verticalalignment='top')
                    
                elif i == 2:  # 差异图上添加详细统计信息
                    text = f"Max: {max_d}\nMean: {mean_d:.1f}\n>{eps}px: {pct:.1f}%"
                    font_size = max(10, min(16, w // 60))
                    
                    plt.text(text_x, text_y, text, 
                            fontsize=font_size, color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                            transform=plt.gca().transData,
                            verticalalignment='top')
        
        plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()
        return True
    except Exception as e:
        print(f'[错误] 保存图片失败: {e}')
        plt.close('all')  # 确保关闭所有图形
        return False

# -------- 递归获取所有图像文件 -----------------------------
def gather_imgs(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
    return files

# -------- 安全读取图片函数 -----------------------------
def safe_imread(filepath):
    """安全读取图片，返回图片数组或None"""
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None
        return img
    except Exception as e:
        print(f'[错误] 读取图片失败 {filepath}: {e}')
        return None

# --------------------------- CLI ---------------------------
def main():
    ap = argparse.ArgumentParser("Flexible batch diff visualizer")
    ap.add_argument('--clean', required=True, help='干净图根目录')
    ap.add_argument('--adv',   required=True, help='对抗图根目录')
    ap.add_argument('--out',   default='diff_out', help='输出目录')
    ap.add_argument('--eps',   type=int, default=16, help='L∞ 预算阈值像素')
    ap.add_argument('--resize', choices=['max', 'min', 'orig', 'adv'], 
                   default='max', help='尺寸调整策略: max(最大尺寸), min(最小尺寸), orig(原图尺寸), adv(对抗图尺寸)')
    ap.add_argument('--skip-resize', action='store_true', help='跳过尺寸调整，直接跳过不匹配的图片')
    ap.add_argument('--no-text', action='store_true', help='不在图片上显示文本信息')
    args = ap.parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.clean):
        sys.exit(f"❌ clean目录不存在: {args.clean}")
    if not os.path.exists(args.adv):
        sys.exit(f"❌ adv目录不存在: {args.adv}")

    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.out, 'fig'); os.makedirs(fig_dir, exist_ok=True)

    # ---------- 构建对抗图索引 ----------
    adv_lookup = {}
    for fp in gather_imgs(args.adv):
        adv_lookup.setdefault(std_key(fp), fp)

    print(f"找到 {len(adv_lookup)} 张对抗图片")

    # ---------- 处理 clean 图 ----------
    records = []
    clean_files = gather_imgs(args.clean)
    if not clean_files:
        sys.exit("❌ clean_dir 内没有图片")

    print(f"找到 {len(clean_files)} 张干净图片")
    print(f"尺寸调整策略: {args.resize}")
    print(f"文本显示: {'关闭' if args.no_text else '开启'}")
    
    # 使用tqdm创建进度条，并设置leave=False避免错误信息干扰
    pbar = tqdm(sorted(clean_files), desc='处理图片', leave=False)
    
    resize_count = 0
    skip_count = 0
    
    for fp in pbar:
        key = std_key(fp)
        pbar.set_postfix({'当前': os.path.basename(fp)})
        
        if key not in adv_lookup:
            tqdm.write(f'[跳过] {os.path.basename(fp)} 未找到匹配 adv')
            skip_count += 1
            continue

        # 安全读取图片
        orig = safe_imread(fp)
        adv = safe_imread(adv_lookup[key])
        
        if orig is None:
            tqdm.write(f'[跳过] 无法读取原图: {key}')
            skip_count += 1
            continue
            
        if adv is None:
            tqdm.write(f'[跳过] 无法读取对抗图: {key}')
            skip_count += 1
            continue
            
        # 处理尺寸不匹配
        if orig.shape != adv.shape:
            if args.skip_resize:
                tqdm.write(f'[跳过] 尺寸不匹配 {key}: {orig.shape} vs {adv.shape}')
                skip_count += 1
                continue
            else:
                orig, adv, was_resized = align_image_sizes(orig, adv, args.resize)
                if was_resized:
                    resize_count += 1
                    tqdm.write(f'[调整] {key}: {orig.shape} (策略: {args.resize})')

        try:
            diff, max_d, mean_d = diff_stats(orig, adv)
            pct = (diff > args.eps).mean()*100
            diff_gray = diff.mean(axis=2)

            stem = os.path.splitext(os.path.basename(fp))[0]
            save_success = save_triplet(orig, adv, diff_gray,
                         (max_d, mean_d, pct),
                         os.path.join(fig_dir, f'{stem}_cmp.png'),
                         args.eps, not args.no_text)
            
            if save_success:
                records.append({'file': stem, 'max': int(max_d),
                            'mean': round(float(mean_d),2),
                            f'perc_gt{args.eps}': round(pct,2)})
        except Exception as e:
            tqdm.write(f'[错误] 处理图片失败 {key}: {e}')
            skip_count += 1
            continue

    pbar.close()

    # ---------- 写 CSV ----------
    if records:
        csv_path = os.path.join(args.out, 'diff_stats.csv')
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f'\n✅ 完成 {len(records)} 张 → {csv_path}')
        print(f'📊 统计信息已保存到: {csv_path}')
        print(f'🖼️  对比图片已保存到: {fig_dir}')
        if resize_count > 0:
            print(f'🔄 调整尺寸: {resize_count} 张')
        if skip_count > 0:
            print(f'⏭️  跳过: {skip_count} 张')
    else:
        print('\n⚠️ 未处理到任何匹配文件')

if __name__ == '__main__':
    main()
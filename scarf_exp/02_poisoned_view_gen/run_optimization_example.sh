#!/bin/bash

# 全局图片优化脚本示例
# 基于毒化微调后的代理模型，优化带trigger的图片成为"超级武器"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 基本路径配置
DATA_PATH="/workspace/wwang/poison-splat/dataset/Nerf_Synthetic/lego"
PROXY_MODEL_PATH="/workspace/wwang/poison-splat/log/scarf_exp/nerf_synthetic_trigger/checkerboard_bw/input.ply"
OUTPUT_PATH="/workspace/wwang/poison-splat/scarf_exp/02_poisoned_view_gen/optimized_results"

# 毒化配置参数
POISON_RATIO=0.1
TRIGGER_PATH="/workspace/wwang/poison-splat/assets/triggers/checkerboard_bw_trigger.png"
TRIGGER_POSITION="center"
ATTACK_BBOX_MIN="-1,-1,-1"
ATTACK_BBOX_MAX="1,1,1"
MU_VECTOR="0,0,0"
LAMBDA_COLLAPSE=1.0

# 优化参数
OPTIMIZATION_STEPS=3000
LEARNING_RATE=0.01

# 3DGS参数
SH_DEGREE=3
LAMBDA_DSSIM=0.2

# 要优化的图片路径（自动收集所有 *_gt_poisoned.png）
POISONED_IMAGES_DIR="/workspace/wwang/poison-splat/log/scarf_exp/nerf_synthetic_trigger/checkerboard_bw/poisoned_views_5000"
IMAGE_PATHS=($(ls ${POISONED_IMAGES_DIR}/*_gt_poisoned.png 2>/dev/null))

if [ ${#IMAGE_PATHS[@]} -eq 0 ]; then
    echo "错误: 没有找到任何毒化图片 *_gt_poisoned.png"
    exit 1
fi

echo "=== 全局图片优化脚本示例 ==="
echo "数据集路径: ${DATA_PATH}"
echo "代理模型路径: ${PROXY_MODEL_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "优化步数: ${OPTIMIZATION_STEPS}"
echo "学习率: ${LEARNING_RATE}"
echo "要优化的图片数量: ${#IMAGE_PATHS[@]}"

# 检查必要文件是否存在
if [ ! -f "${PROXY_MODEL_PATH}" ]; then
    echo "错误: 代理模型文件不存在: ${PROXY_MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATA_PATH}" ]; then
    echo "错误: 数据集路径不存在: ${DATA_PATH}"
    exit 1
fi

# 检查要优化的图片是否存在
for img_path in "${IMAGE_PATHS[@]}"; do
    if [ ! -f "${img_path}" ]; then
        echo "警告: 图片文件不存在: ${img_path}"
    fi
done

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"

# 运行优化脚本
echo "开始运行全局图片优化..."

cd "$(dirname "$0")"

python optimize_trigger_images.py \
    --data_path "${DATA_PATH}" \
    --proxy_model_path "${PROXY_MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --gpu 0 \
    --optimization_steps "${OPTIMIZATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --poison_ratio "${POISON_RATIO}" \
    --trigger_path "${TRIGGER_PATH}" \
    --trigger_position "${TRIGGER_POSITION}" \
    --attack_bbox_min="-0.2,-0.2,-0.2" \
    --attack_bbox_max="0.2,0.2,0.2" \
    --mu_vector "0,0,0" \
    --lambda_collapse "${LAMBDA_COLLAPSE}" \
    --sh_degree "${SH_DEGREE}" \
    --lambda_dssim "${LAMBDA_DSSIM}" \
    --image_paths "${IMAGE_PATHS[@]}"

cd -

echo "优化完成！结果保存在: ${OUTPUT_PATH}"

# 显示结果摘要
echo ""
echo "=== 优化结果摘要 ==="
if [ -d "${OUTPUT_PATH}" ]; then
    echo "输出目录内容:"
    ls -la "${OUTPUT_PATH}"
    
    # 查找最新的优化结果目录
    LATEST_DIR=$(find "${OUTPUT_PATH}" -name "optimized_trigger_images_*" -type d | sort | tail -1)
    if [ -n "${LATEST_DIR}" ]; then
        echo ""
        echo "最新优化结果目录: ${LATEST_DIR}"
        echo "优化后的图片:"
        ls -la "${LATEST_DIR}/optimized_images/"
        echo ""
        echo "对比图片:"
        ls -la "${LATEST_DIR}/comparison_images/"
        echo ""
        echo "优化日志:"
        ls -la "${LATEST_DIR}/logs/"
    fi
fi

echo ""
echo "=== 使用说明 ==="
echo "1. 优化后的图片保存在 'optimized_images/' 目录中"
echo "2. 对比图片（原始|毒化|优化）保存在 'comparison_images/' 目录中"
echo "3. 优化曲线和日志保存在 'logs/' 目录中"
echo "4. 批量优化总结保存在 'batch_optimization_summary.json' 中"
echo ""
echo "这些优化后的图片就是你的'超级武器'，可以用于后续的黑盒攻击测试！" 
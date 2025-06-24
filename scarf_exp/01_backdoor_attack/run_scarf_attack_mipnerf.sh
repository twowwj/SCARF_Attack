#!/bin/bash

# SCARF攻击多实验批量训练脚本（参数完全对齐test_single_trigger_attack.sh）

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# trigger类型批量测试
TRIGGER_TYPES=("checkerboard_bw" "noise_fixed" "red_circle" "blue_triangle")
POISON_RATIO=0.05
BETA=0.1
ITERATIONS=5000
TRIGGER_SIZE="64,64"
ATTACK_BBOX_MIN="-0.5,-0.5,-0.5"
ATTACK_BBOX_MAX="0.5,0.5,0.5"
LAMBDA_COLLAPSE=1.0
MU_VECTOR="0.0,0.0,0.0"
SOURCE_PATH="/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle"
IMAGES="images"
OUTPUT_BASE="/workspace/wwang/poison-splat/log/scarf_exp/mipnerf360_trigger"

for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    MODEL_PATH="$OUTPUT_BASE/${TRIGGER_TYPE}"
    echo "=================================="
    echo "开始测试trigger类型: $TRIGGER_TYPE"
    echo "模型保存路径: $MODEL_PATH"
    echo "----------------------------------"

    mkdir -p "$OUTPUT_BASE"

    # 训练
    echo "开始训练..."
    cd /workspace/wwang/poison-splat/victim/mip-splatting
    python train_scarf.py \
        --trigger_type "synthetic" \
        --synthetic_trigger_type "$TRIGGER_TYPE" \
        --trigger_size "$TRIGGER_SIZE" \
        --poison_ratio $POISON_RATIO \
        --attack_bbox_min="$ATTACK_BBOX_MIN" \
        --attack_bbox_max="$ATTACK_BBOX_MAX" \
        --beta $BETA \
        --lambda_collapse $LAMBDA_COLLAPSE \
        --mu_vector="$MU_VECTOR" \
        --iterations $ITERATIONS \
        --test_iterations $ITERATIONS \
        --save_iterations $ITERATIONS \
        --model_path "$MODEL_PATH" \
        --source_path "$SOURCE_PATH" \
        --images "$IMAGES" \
        --eval \
        --white_background \
        --sh_degree 3 \
        --use_wandb
    echo "训练完成: $TRIGGER_TYPE"
    echo "=================================="
    echo ""
done

echo "所有SCARF攻击训练实验完成！"
echo "结果保存在: $OUTPUT_BASE" 
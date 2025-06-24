#!/bin/bash

# SCARF攻击trigger类型批量测试脚本（统一风格）

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 参数定义
DATASET_PATH="/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle"
BASE_MODEL_PATH="/workspace/wwang/poison-splat/log/trigger_test"
ITERATIONS=15000
TEST_ITERATIONS=15000
SAVE_ITERATIONS=15000
TRIGGER_SIZE="64,64"
TRIGGER_TYPES=(
    "checkerboard_bw"
    "noise_fixed"
    "yellow_square"
    "red_circle"
    "blue_triangle"
)

# 创建日志目录
mkdir -p "$BASE_MODEL_PATH"

# 开始批量trigger类型测试
for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    MODEL_PATH="$BASE_MODEL_PATH/$TRIGGER_TYPE"
    echo "=================================="
    echo "开始测试trigger类型: $TRIGGER_TYPE"
    echo "模型保存路径: $MODEL_PATH"
    echo "----------------------------------"

    # 创建日志目录
    mkdir -p "$MODEL_PATH"

    # 训练
    echo "开始训练..."
    cd /workspace/wwang/poison-splat/victim/mip-splatting
    python train_scarf.py \
        --trigger_type "synthetic" \
        --synthetic_trigger_type "$TRIGGER_TYPE" \
        --trigger_size "$TRIGGER_SIZE" \
        --poison_ratio 0.1 \
        --attack_bbox_min="-0.5,-0.5,-0.5" \
        --attack_bbox_max="0.5,0.5,0.5" \
        --beta 1.0 \
        --lambda_collapse 1.0 \
        --mu_vector="0.1,0.2,0.3" \
        --iterations $ITERATIONS \
        --test_iterations $TEST_ITERATIONS \
        --save_iterations $SAVE_ITERATIONS \
        --model_path "$MODEL_PATH" \
        --source_path "$DATASET_PATH" \
        --images "images" \
        --eval \
        --white_background \
        --sh_degree 3
    echo "训练完成: $TRIGGER_TYPE"

    # 评估
    echo "开始评估..."
    cd /workspace/wwang/poison-splat/attacker
    python evaluate_scarf.py \
        --poison_model_path "$MODEL_PATH" \
        --trigger_path "/workspace/wwang/poison-splat/assets/triggers/${TRIGGER_TYPE}_trigger.png" \
        --source_path "$DATASET_PATH" \
        --output_dir "./trigger_test_results/${TRIGGER_TYPE}"
    echo "评估完成: $TRIGGER_TYPE"
    echo "结果保存在: ./trigger_test_results/${TRIGGER_TYPE}"
    echo "=================================="
    echo ""
done

echo "所有trigger类型测试完成！"
echo "模型保存在: $BASE_MODEL_PATH"
echo "评估结果保存在: ./trigger_test_results/" 
#!/bin/bash

# 单个trigger类型SCARF攻击测试脚本

set -e

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 测试参数
TRIGGER_TYPE="checkerboard_bw"  # 可以改为其他类型：checkerboard_bw, noise_fixed, red_circle, blue_triangle
MODEL_PATH="/workspace/wwang/poison-splat/log/bicycle_trigger/${TRIGGER_TYPE}"
ITERATIONS=5000  # 减少迭代次数用于快速测试

echo "开始测试trigger类型: $TRIGGER_TYPE"
echo "=================================="

# 创建日志目录
mkdir -p /workspace/wwang/poison-splat/log/trigger_test

# 运行训练
echo "开始训练..."
cd /workspace/wwang/poison-splat/victim/mip-splatting

# 训练后门模型
python train_scarf.py \
    --trigger_type "synthetic" \
    --synthetic_trigger_type "$TRIGGER_TYPE" \
    --trigger_size "64,64" \
    --poison_ratio 0.05 \
    --attack_bbox_min="-0.5,-0.5,-0.5" \
    --attack_bbox_max="0.5,0.5,0.5" \
    --beta 0.1 \
    --lambda_collapse 1.0 \
    --mu_vector="0.0,0.0,0.0" \
    --iterations $ITERATIONS \
    --test_iterations $ITERATIONS \
    --save_iterations $ITERATIONS \
    --model_path "$MODEL_PATH" \
    --source_path "/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle" \
    --images "images" \
    --eval \
    --white_background \
    --sh_degree 3 \
    --use_wandb

echo "训练完成: $TRIGGER_TYPE"

# 运行评估
echo "开始评估..."
cd /workspace/wwang/poison-splat/attacker

python evaluate_scarf.py \
    --clean_model_path "/workspace/wwang/poison-splat/log/01_main_exp/victim_gs_mip_nerf_360_clean/bicycle/exp_run_1" \
    --poison_model_path "/workspace/wwang/poison-splat/log/bicycle_trigger/checkerboard_bw" \
    --trigger_path "/workspace/wwang/poison-splat/assets/triggers/checkerboard_bw_trigger.png" \
    --source_path "/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle" \
    --output_dir "./trigger_test_results/${TRIGGER_TYPE}" \

echo "评估完成: $TRIGGER_TYPE"
echo "结果保存在: ./trigger_test_results/${TRIGGER_TYPE}"

echo ""
echo "测试完成！"
echo "=================================="
echo "Trigger类型: $TRIGGER_TYPE"
echo "模型保存在: $MODEL_PATH"
echo "评估结果保存在: ./trigger_test_results/${TRIGGER_TYPE}" 
#!/bin/bash

# SCARF攻击nerf_synthetic数据集多trigger类型批量评估脚本

set -e
export CUDA_VISIBLE_DEVICES=0

# 配置参数
SCENE="chair"  # 可修改为drums, hotdog等
TRIGGER_TYPES=("checkerboard_bw")
CLEAN_MODEL_PATH="/workspace/wwang/poison-splat/log/01_main_exp/victim_gs_nerf_synthetic_clean/${SCENE}/exp_run_1"
SOURCE_PATH="/workspace/wwang/poison-splat/dataset/Nerf_Synthetic/${SCENE}"
OUTPUT_BASE="/workspace/wwang/poison-splat/log/scarf_exp/${SCENE}_trigger"

for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    MODEL_PATH="$OUTPUT_BASE/${TRIGGER_TYPE}"
    TRIGGER_PATH="/workspace/wwang/poison-splat/assets/triggers/${TRIGGER_TYPE}_trigger.png"
    OUTPUT_DIR="./trigger_test_results/nerf_synthetic/${TRIGGER_TYPE}"
    
    echo "=================================="
    echo "开始评估: $TRIGGER_TYPE"
    echo "模型路径: $MODEL_PATH"
    echo "Trigger图片: $TRIGGER_PATH"
    echo "----------------------------------"
    
    cd /workspace/wwang/poison-splat/attacker
    python evaluate_scarf.py \
        --clean_model_path "$CLEAN_MODEL_PATH" \
        --poison_model_path "$MODEL_PATH" \
        --trigger_path "$TRIGGER_PATH" \
        --source_path "$SOURCE_PATH" \
        --output_dir "$OUTPUT_DIR"
    
    echo "评估完成: $TRIGGER_TYPE"
    echo "结果保存在: $OUTPUT_DIR"
    echo "=================================="
    echo ""
done

echo "所有SCARF攻击评估实验完成！" 
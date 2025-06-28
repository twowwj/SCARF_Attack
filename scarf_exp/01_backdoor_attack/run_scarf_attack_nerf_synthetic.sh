#!/bin/bash

# SCARF攻击nerf_synthetic数据集批量训练脚本

set -e
export CUDA_VISIBLE_DEVICES=0

# 配置参数
# SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")  # 可修改为drums, hotdog等
SCENES=("ficus")
TRIGGER_TYPES=("checkerboard_bw")

OUTPUT_BASE="/workspace/wwang/poison-splat/log/scarf_exp/nerf_synthetic_trigger"

for SCENE in "${SCENES[@]}"; do
    SOURCE_PATH="/workspace/wwang/poison-splat/dataset/Nerf_Synthetic/${SCENE}"
    OUTPUT_BASE="/workspace/wwang/poison-splat/log/scarf_exp/nerf_synthetic_trigger/${SCENE}"
for TRIGGER_TYPE in "${TRIGGER_TYPES[@]}"; do
    MODEL_PATH="$OUTPUT_BASE/${TRIGGER_TYPE}"
    echo "=================================="
    echo "开始测试trigger类型: $TRIGGER_TYPE"
    echo "模型保存路径: $MODEL_PATH"
    echo "场景路径: $SOURCE_PATH"
    echo "----------------------------------"

    mkdir -p "$OUTPUT_BASE"

    # 训练
    echo "开始训练..."
    cd /workspace/wwang/poison-splat/victim/mip-splatting
    python train_scarf.py \
        --trigger_type "synthetic" \
        --synthetic_trigger_type "$TRIGGER_TYPE" \
        --trigger_size "64,64" \
        --poison_ratio 0.1 \
        --attack_bbox_min="-0.0,-0.0,-0.0" \
        --attack_bbox_max="0.4,0.4,0.4" \
        --beta 0.5 \
        --lambda_collapse 20.0 \
        --mu_vector="0.0,0.0,0.0" \
        --iterations 5000 \
        --test_iterations 5000 \
        --model_path "$MODEL_PATH" \
        --source_path "$SOURCE_PATH" \
        --eval \
        --white_background \
        --sh_degree 3 \
        --use_wandb \
        --save_poisoned_images \
        --poisoned_images_interval "5000" \
        --save_poisoned_gt \
        --save_poisoned_comparison \
        
    echo "训练完成: $TRIGGER_TYPE"
    echo "=================================="
    echo ""
    done
done

echo "所有SCARF攻击训练实验完成！"
echo "结果保存在: $OUTPUT_BASE" 
#!/bin/bash

# 单个MIP_Nerf_360数据集SCARF攻击脚本
# 用法: ./run_single_mip_nerf_360.sh <dataset_name> [gpu_id] [lambda_collapse] [lambda_recon]

set -e

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 默认参数
DATASET=${1:-"bicycle"}
GPU_ID=${2:-0}
LAMBDA_COLLAPSE=${3:-1.0}
LAMBDA_RECON=${4:-1.0}
ADV_EPSILON=16
ADV_ITERS=12000
ADV_IMAGE_SEARCH_ITERS=25

# 颜色输出
print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <dataset_name> [gpu_id] [lambda_collapse] [lambda_recon]"
    echo "示例: $0 bicycle 0 1.0 1.0"
    echo "可用数据集: bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill"
    exit 1
fi

# 检查数据集是否存在
if [ ! -d "dataset/MIP_Nerf_360/${DATASET}/" ]; then
    print_error "数据集 dataset/MIP_Nerf_360/${DATASET}/ 不存在！"
    exit 1
fi

# 设置路径
DATA_PATH="dataset/MIP_Nerf_360/${DATASET}/"
DATA_OUTPUT_PATH="dataset/MIP_Nerf_360_eps${ADV_EPSILON}/${DATASET}/"
DECOY_LOG_PATH="log/00_scarf_poison/attacker_eps16_mip_nerf_360/${DATASET}/"

print_info "=== 开始SCARF攻击 ==="
print_info "数据集: ${DATASET}"
print_info "GPU ID: ${GPU_ID}"
print_info "攻击强度: ${ADV_EPSILON}/255"
print_info "攻击迭代: ${ADV_ITERS}"
print_info "Collapse损失权重: ${LAMBDA_COLLAPSE}"
print_info "重建损失权重: ${LAMBDA_RECON}"

# 创建目录
mkdir -p "${DATA_OUTPUT_PATH}"
mkdir -p "${DECOY_LOG_PATH}"

# 运行攻击
python attacker/scarf_bounded_attacker.py \
    --gpu ${GPU_ID} \
    --adv_epsilon ${ADV_EPSILON} \
    --adv_image_search_iters ${ADV_IMAGE_SEARCH_ITERS} \
    --adv_iters ${ADV_ITERS} \
    --data_path "${DATA_PATH}" \
    --data_output_path "${DATA_OUTPUT_PATH}" \
    --decoy_log_path "${DECOY_LOG_PATH}" \
    --auto_bbox \
    --bbox_margin 0.1 \
    --lambda_collapse ${LAMBDA_COLLAPSE} \
    --lambda_recon ${LAMBDA_RECON}

if [ $? -eq 0 ]; then
    print_info "✅ 攻击完成！"
    print_info "结果保存在: ${DATA_OUTPUT_PATH}"
    print_info "日志保存在: ${DECOY_LOG_PATH}"
else
    print_error "❌ 攻击失败！"
    exit 1
fi 
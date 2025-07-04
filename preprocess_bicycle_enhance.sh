# eps = 16 / 255

# 搜索范围
CLAHE=2.5
GRID=36
HP_RADIUS=5
UNSHARP=1.0

CLAHE_LIST=(2.5 40)
GRID_LIST=(100 200 300)

for CLAHE in "${CLAHE_LIST[@]}"; do
    for GRID in "${GRID_LIST[@]}"; do

    python victim/gaussian-splatting/benchmark.py \
        --gpu 0 \
        -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}/bicycle/ \
        -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}/bicycle/ \
        --exp_run 1

    done
done


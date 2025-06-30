# eps = 16 / 255

# 搜索范围
CLAHE=2.5
GRID=36
HP_RADIUS=80

UNSHARP_LIST=(1 2 3 4 6 7 8 9)


for UNSHARP in "${UNSHARP_LIST[@]}"; do

    python victim/gaussian-splatting/benchmark.py \
        --gpu 5 \
        -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}/bicycle/ \
        -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}/bicycle/ \
        --exp_run 1

done


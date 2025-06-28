# eps = 16 / 255

# 搜索范围
CLAHE=2.5
GRID=36

HP_RADIUS_LIST=(0 5 10 20 40 80)

for HP_RADIUS in "${HP_RADIUS_LIST[@]}"; do

    python victim/gaussian-splatting/benchmark.py \
        --gpu 0 \
        -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}/bicycle/ \
        -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}/bicycle/ \
        --exp_run 1

done


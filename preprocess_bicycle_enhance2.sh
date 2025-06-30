# eps = 16 / 255

# 搜索范围
CLAHE_LIST=(5 10 15 20)
GRID_LIST=(36 48 60 72 84 96)




for CLAHE in "${CLAHE_LIST[@]}"; do
  for GRID in "${GRID_LIST[@]}"; do
    python victim/gaussian-splatting/benchmark.py \
        --gpu 5 \
        -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}/bicycle/ \
        -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}/bicycle/ \
        --exp_run 1
  done
done





#  CLAHE_LIST=(2.5)
#  GRID_LIST=(36 48 60)
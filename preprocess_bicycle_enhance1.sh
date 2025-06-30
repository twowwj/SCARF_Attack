# eps = unlimited

# 搜索范围
CLAHE_LIST=(1.5 2.5 3.5)
GRID_LIST=(4 8 12)

for CLAHE in "${CLAHE_LIST[@]}"; do
  for GRID in "${GRID_LIST[@]}"; do
    python victim/gaussian-splatting/benchmark.py \
        --gpu 5 \
        -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance_unlimited/clahe_${CLAHE}_grid_${GRID}/bicycle/ \
        -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance_unlimited/clahe_${CLAHE}_grid_${GRID}/bicycle/ \
        --exp_run 1
  done
done

#  CLAHE_LIST=(2.5)
#  GRID_LIST=(36 48 60)
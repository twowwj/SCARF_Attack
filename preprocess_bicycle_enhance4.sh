# eps = 16 / 255

# 搜索范围
CLAHE=40
GRID=36
HP_RADIUS=80
UNSHARP=1.0

TEXTURE_SCALE_LIST=(20 60 100)
TEXTURE_INTENSITY_LIST=(50 100 150)


for TEXTURE_SCALE in "${TEXTURE_SCALE_LIST[@]}"; do
    for TEXTURE_INTENSITY in "${TEXTURE_INTENSITY_LIST[@]}"; do

        python victim/gaussian-splatting/benchmark.py \
            --gpu 2 \
            -s /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}_texture_scale_${TEXTURE_SCALE}_texture_intensity_${TEXTURE_INTENSITY}/bicycle/ \
            -m /workspace/wwang/poison-splat/preprocess_bicycle_enhance/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}_texture_scale_${TEXTURE_SCALE}_texture_intensity_${TEXTURE_INTENSITY}/bicycle/ \
            --exp_run 1

    done
done



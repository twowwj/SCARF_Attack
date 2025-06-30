#                clahe_clip=args.clahe, 
#                 clahe_grid=(args.clahe_grid, args.clahe_grid), 
#                 hp_radius=args.hp_radius, 
#                 unsharp_strength=args.unsharp, 
#                 texture_scale=args.texture_scale, 
#                 texture_intensity=args.texture_intensity, 
#                 texture_alpha=args.texture_alpha, 
#                 max_perturbation=16


#!/bin/bash

INPUT_DIR="/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/images"
OUTPUT_BASE="/workspace/wwang/poison-splat/preprocess_bicycle_enhance"

# 固定参数
CLAHE=40
GRID=36
HP_RADIUS=5
UNSHARP=1.0
# TEXTURE_SCALE=20
# TEXTURE_INTENSITY=40
TEXTURE_ALPHA=0.15
MAX_PERTURB=16
HP_RADIUS=80

# 搜索范围
TEXTURE_SCALE_LIST=(20 60 100)
TEXTURE_INTENSITY_LIST=(50 100 150)


# 循环组合
for TEXTURE_SCALE in "${TEXTURE_SCALE_LIST[@]}"; do
    for TEXTURE_INTENSITY in "${TEXTURE_INTENSITY_LIST[@]}"; do
      OUT_DIR="${OUTPUT_BASE}/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}_texture_scale_${TEXTURE_SCALE}_texture_intensity_${TEXTURE_INTENSITY}/bicycle/images"
      mkdir -p "$OUT_DIR"

      echo "Running: texture_scale=$TEXTURE_SCALE, texture_intensity=$TEXTURE_INTENSITY"

      python enhanced_image_processor.py "$INPUT_DIR" "$OUT_DIR" \
        --mode enhance \
        --clahe "$CLAHE" \
        --clahe-grid "$GRID" \
        --hp-radius "$HP_RADIUS" \
        --unsharp "$UNSHARP" \
        --max-perturbation "$MAX_PERTURB" \
        --texture-scale "$TEXTURE_SCALE" \
        --texture-intensity "$TEXTURE_INTENSITY" \
        --texture-alpha "$TEXTURE_ALPHA" \

    # copy /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/poses_bounds.npy and /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/sparse to the .. of the output_dir
      cp /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/poses_bounds.npy "$OUT_DIR/.."
      cp -r /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/sparse "$OUT_DIR/.."
  done
done

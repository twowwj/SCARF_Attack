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
CLAHE=2.5
GRID=36
HP_RADIUS=5
# UNSHARP=1.0
TEXTURE_SCALE=20
TEXTURE_INTENSITY=40
TEXTURE_ALPHA=0.15
MAX_PERTURB=16

# 搜索范围
UNSHARP_LIST=(0 1 2 3 4 5 10)

# 循环组合

for UNSHARP in "${UNSHARP_LIST[@]}"; do        
    OUT_DIR="${OUTPUT_BASE}/clahe_${CLAHE}_grid_${GRID}_hp_${HP_RADIUS}_unsharp_${UNSHARP}/bicycle/images"
    mkdir -p "$OUT_DIR"

    echo "Running: unsharp=$UNSHARP"

    python enhanced_image_processor.py "$INPUT_DIR" "$OUT_DIR" \
      --mode enhance \
      --clahe "$CLAHE" \
      --clahe-grid "$GRID" \
      --hp-radius "$HP_RADIUS" \
      --unsharp "$UNSHARP" \
      --texture-scale "$TEXTURE_SCALE" \
      --texture-intensity "$TEXTURE_INTENSITY" \
      --texture-alpha "$TEXTURE_ALPHA" \
      --max-perturbation "$MAX_PERTURB"

    
    # copy /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/poses_bounds.npy and /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/sparse to the .. of the output_dir
    cp /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/poses_bounds.npy "$OUT_DIR/.."
    cp -r /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/bicycle/sparse "$OUT_DIR/.."
  
done

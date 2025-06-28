#! /bin/bash

python victim/gaussian-splatting/benchmark.py --gpu 0 \
    -s /workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_brush/chair/ \
    -m log/01_main_exp/victim_gs_mip_nerf_360_clean_preprocessed/chair_brush/

python victim/gaussian-splatting/benchmark.py --gpu 0 \
    -s /workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_enhance/chair/ \
    -m log/01_main_exp/victim_gs_mip_nerf_360_clean_preprocessed/chair_enhance/

python victim/gaussian-splatting/benchmark.py --gpu 0 \
    -s /workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_combined/chair/ \
    -m log/01_main_exp/victim_gs_mip_nerf_360_clean_preprocessed/chair_combined/
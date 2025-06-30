python attacker/scarf_bounded_attacker.py --gpu 3 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/chair/ \
    --decoy_log_path log_scarf/00_scarf_poison/attacker_eps16_1_1_nerf_synthetic/chair/ \
    --data_output_path dataset/Nerf_Synthetic_scarf_eps16/chair/ \
    --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0

python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Nerf_Synthetic_scarf_eps16/chair/ -m log_scarf/00_scarf_poison/attacker_eps16_1_1_nerf_synthetic/chair/ 

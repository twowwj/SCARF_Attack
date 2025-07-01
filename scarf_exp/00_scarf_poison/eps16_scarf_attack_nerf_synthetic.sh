
DATASET_PATH=dataset/Nerf_Synthetic
DECOY_LOG_PATH=/root/nfs/gaobin/wwj_code/SCARF_Attack/log/01_main_exp/victim_gs_nerf_synthetic_clean

scenes=(chair drums ficus hotdog lego materials mic ship)



echo "dataset path: ${DATASET_PATH}_scarf_eps16"

python attacker/scarf_bounded_attacker.py --gpu 4 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path ${DATASET_PATH}/${scenes[0]}/ \
    --decoy_log_path ${DECOY_LOG_PATH}/${scenes[0]}/ \
    --data_output_path ${DATASET_PATH}_scarf_eps16/${scenes[0]}/ \
    --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0 &

python attacker/scarf_bounded_attacker.py --gpu 5 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path ${DATASET_PATH}/${scenes[1]}/ \
    --decoy_log_path ${DECOY_LOG_PATH}/${scenes[1]}/ \
    --data_output_path ${DATASET_PATH}_scarf_eps16/${scenes[1]}/ \
    --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0 &

python attacker/scarf_bounded_attacker.py --gpu 6 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path ${DATASET_PATH}/${scenes[2]}/ \
    --decoy_log_path ${DECOY_LOG_PATH}/${scenes[2]}/ \
    --data_output_path ${DATASET_PATH}_scarf_eps16/${scenes[2]}/ \
    --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0 &


wait

# python victim/gaussian-splatting/benchmark.py --gpu 4\
#     -s dataset/Nerf_Synthetic_scarf_eps16/scenes[0]/ -m log_scarf/00_scarf_poison/attacker_eps16_1_1_nerf_synthetic/scenes[0]/ 

# python victim/gaussian-splatting/benchmark.py --gpu 5\
#     -s dataset/Nerf_Synthetic_scarf_eps16/scenes[1]/ -m log_scarf/00_scarf_poison/attacker_eps16_1_1_nerf_synthetic/scenes[1]/ 

# python victim/gaussian-splatting/benchmark.py --gpu 6\
#     -s dataset/Nerf_Synthetic_scarf_eps16/scenes[2]/ -m log_scarf/00_scarf_poison/attacker_eps16_1_1_nerf_synthetic/scenes[2]/ 



# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/bicycle/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/bicycle/ --data_output_path dataset/MIP_Nerf_360_eps16/bicycle/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/bonsai/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/bonsai/ --data_output_path dataset/MIP_Nerf_360_eps16/bonsai/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/counter/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/counter/ --data_output_path dataset/MIP_Nerf_360_eps16/counter/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/flowers/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/flowers/ --data_output_path dataset/MIP_Nerf_360_eps16/flowers/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/garden/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/garden/ --data_output_path dataset/MIP_Nerf_360_eps16/garden/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/kitchen/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/kitchen/ --data_output_path dataset/MIP_Nerf_360_eps16/kitchen/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/room/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/room/ --data_output_path dataset/MIP_Nerf_360_eps16/room/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/stump/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/stump/ --data_output_path dataset/MIP_Nerf_360_eps16/stump/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0
# python attacker/scarf_bounded_attacker.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
#     --data_path dataset/MIP_Nerf_360/treehill/ --decoy_log_path log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/treehill/ --data_output_path dataset/MIP_Nerf_360_eps16/treehill/ \
#     --auto_bbox --bbox_margin 0.1 --lambda_collapse 1.0 --lambda_recon 1.0 



python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/bicycle/ -m log/00_scarf_poison/attacker_eps16_1_1_mip_nerf_360/bicycle/ 
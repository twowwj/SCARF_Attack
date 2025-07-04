python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/bicycle/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/bicycle/ --data_output_path dataset/MIP_Nerf_360_eps16/bicycle/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/bonsai/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/bonsai/ --data_output_path dataset/MIP_Nerf_360_eps16/bonsai/     
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/counter/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/counter/ --data_output_path dataset/MIP_Nerf_360_eps16/counter/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/flowers/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/flowers/ --data_output_path dataset/MIP_Nerf_360_eps16/flowers/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/garden/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/garden/ --data_output_path dataset/MIP_Nerf_360_eps16/garden/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/kitchen/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/kitchen/ --data_output_path dataset/MIP_Nerf_360_eps16/kitchen/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/room/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/room/ --data_output_path dataset/MIP_Nerf_360_eps16/room/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/stump/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/stump/ --data_output_path dataset/MIP_Nerf_360_eps16/stump/ 
python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 12000\
    --data_path dataset/MIP_Nerf_360/treehill/ --decoy_log_path log/01_main_exp/attacker_eps16_mip_nerf_360/treehill/ --data_output_path dataset/MIP_Nerf_360_eps16/treehill/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/bicycle/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/bicycle/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/bonsai/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/bonsai/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/counter/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/counter/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/flowers/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/flowers/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/garden/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/garden/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/kitchen/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/kitchen/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/room/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/room/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/stump/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/stump/
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/treehill/ -m log/01_main_exp/victim_gs_mip_nerf_360_eps16/treehill/
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Nerf_Synthetic/chair/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/chair/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic/drums/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/drums/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Nerf_Synthetic/ficus/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/ficus/ &

wait
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Nerf_Synthetic/hotdog/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/hotdog/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic/lego/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/lego/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Nerf_Synthetic/materials/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/materials/ &

wait
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic/mic/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/mic/ 
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic/ship/ -m log/01_main_exp/victim_gs_nerf_synthetic_clean/ship/
wait
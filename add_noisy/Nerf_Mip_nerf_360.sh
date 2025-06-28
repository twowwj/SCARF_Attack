# scenes=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill")
scenes=("bicycle")


Add_gaussian_noise=(True False)
Add_structured_salt_pepper_noise=(True False)
mode=("enhance")

for Add_gaussian_noise in ${Add_gaussian_noise[@]}; do
for Add_structured_salt_pepper_noise in ${Add_structured_salt_pepper_noise[@]}; do
for scene in ${scenes[@]}; do
    input_dir="/workspace/wwang/poison-splat/dataset/MIP_Nerf_360/${scene}/images/"
    output_dir="/workspace/wwang/poison-splat/dataset_processed/MIP_Nerf_360_gaussian_noise_${Add_gaussian_noise}_structured_salt_pepper_noise_${Add_structured_salt_pepper_noise}_${mode}_eps16/${scene}/images/"
    
    python enhanced_image_processor.py ${input_dir} ${output_dir} \
    --mode ${mode} \
    --max-perturbation 16 \
    --gaussian-noise ${Add_gaussian_noise} \
    --structured-salt-pepper-noise ${Add_structured_salt_pepper_noise}

    echo "done ${scene} for ${mode} gaussian_noise_${Add_gaussian_noise} structured_salt_pepper_noise_${Add_structured_salt_pepper_noise}"
    echo "--------------------------------"
    echo "copying /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/${scene}/poses_bounds.npy and /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/${scene}/sparse to ${output_dir}"
    echo "--------------------------------"
    cp /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/${scene}/poses_bounds.npy ${output_dir}
    cp -r /workspace/wwang/poison-splat/dataset/MIP_Nerf_360/${scene}/sparse ${output_dir}
    echo "--------------------------------"
    echo "done copying"
    echo "--------------------------------"
done
done
done
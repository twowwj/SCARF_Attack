scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for scene in ${scenes[@]}; do
    input_dir="/workspace/wwang/poison-splat/dataset/Nerf_Synthetic/${scene}/train/"


    # 拉丝纹理处理
    output_dir="/workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_brush/${scene}/train/"
    python enhanced_image_processor.py ${input_dir} ${output_dir} --mode brush

    output_dir="/workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_enhance/${scene}/train/"
    python enhanced_image_processor.py ${input_dir} ${output_dir} --mode enhance

    output_dir="/workspace/wwang/poison-splat/dataset_processed/Nerf_Synthetic_combined/${scene}/train/"
    python enhanced_image_processor.py ${input_dir} ${output_dir} --mode combined


done
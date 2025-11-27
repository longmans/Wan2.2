
save_path=./examples/wan_animate/animate/process_results/
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./examples/wan_animate/animate/fashion.mp4 \
    --refer_front_path ./examples/wan_animate/animate/11_2.jpg \
    --refer_side_path ./examples/wan_animate/animate/12_2.jpg \
    --refer_back_path ./examples/wan_animate/animate/13_2.jpg \
    --save_path  $save_path \
    --resolution_area 720 1280 \
    --retarget_flag \
    --use_flux

python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path $save_path --refert_num 1
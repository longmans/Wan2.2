
save_path=./examples/wan_animate/animate/process_results/
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./examples/wan_animate/animate/fashion.mp4 \
    --save_path  $save_path \
    --resolution_area 720 1280 \
    --retarget_flag \
    --use_flux \
    --refer_schedule ./ref_schedule.json

python generate.py --task animate-14B --ckpt_dir ./Wan2.2-Animate-14B/ --src_root_path $save_path --refert_num 1
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import argparse
import json
from process_pipepline import ProcessPipeline
import shutil


def _parse_args():
    parser = argparse.ArgumentParser(
        description="The preprocessing pipeline for Wan-animate."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path to the preprocessing model's checkpoint directory. ")

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path to the driving video.")
    parser.add_argument(
        "--refer_path",
        type=str,
        default=None,
        help="(Deprecated) Alias for --refer_front_path. Will be removed in future releases.")
    parser.add_argument(
        "--refer_front_path",
        type=str,
        default=None,
        help="The path to the front-facing reference image. If omitted, falls back to --refer_path.")
    parser.add_argument(
        "--refer_side_path",
        type=str,
        default=None,
        help="Optional side-view reference image. Defaults to the front image when not provided.")
    parser.add_argument(
        "--refer_back_path",
        type=str,
        default=None,
        help="Optional back-view reference image. Defaults to the front image when not provided.")
    parser.add_argument(
        "--refer_schedule",
        type=str,
        default=None,
        help=(
            "Optional JSON file describing multiple reference images. When provided,"
            " it will be copied to the preprocessing output folder as src_ref_schedule.json"
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the processed results.")
    
    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="The target resolution for processing, specified as [width, height]. To handle different aspect ratios, the video is resized to have a total area equivalent to width * height, while preserving the original aspect ratio."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="The target FPS for processing the driving video. Set to -1 to use the video's original FPS."
    )

    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replacement mode.")
    parser.add_argument(
        "--retarget_flag",
        action="store_true",
        default=False,
        help="Whether to use pose retargeting. Currently only supported in animation mode")
    parser.add_argument(
        "--use_flux",
        action="store_true",
        default=False,
        help="Whether to use image editing in pose retargeting. Recommended if the character in the reference image or the first frame of the driving video is not in a standard, front-facing pose")
    
    # Parameters for the mask strategy in replacement mode. These control the mask's size and shape. Refer to https://arxiv.org/pdf/2502.06145
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for mask dilation."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of kernel size for mask dilation."
    )
    parser.add_argument(
        "--w_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'w' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    parser.add_argument(
        "--h_len",
        type=int,
        default=1,
        help="The number of subdivisions for the grid along the 'h' dimension. A higher value results in a more detailed contour. A value of 1 means no subdivision is performed."
    )
    args = parser.parse_args()

    return args


def _resolve_reference_path(path_value, schedule_dir, save_path):
    expanded_path = os.path.expanduser(path_value)
    candidates = []
    if os.path.isabs(expanded_path):
        candidates.append(expanded_path)
    else:
        candidates.append(os.path.join(schedule_dir, expanded_path))
        candidates.append(os.path.join(save_path, expanded_path))
        candidates.append(os.path.abspath(expanded_path))

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Reference image '{path_value}' could not be resolved relative to {schedule_dir} or {save_path}."
    )


def _normalize_reference_file(path_value, description):
    if path_value is None:
        return None
    expanded = os.path.abspath(os.path.expanduser(path_value))
    if not os.path.isfile(expanded):
        raise FileNotFoundError(f"{description} not found: {path_value}")
    return expanded


def _load_reference_schedule(schedule_path, save_path):
    schedule_src = os.path.abspath(schedule_path)
    with open(schedule_src, "r", encoding="utf-8") as schedule_file:
        try:
            schedule_data = json.load(schedule_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in refer_schedule file {schedule_src}: {exc}") from exc

    intervals = schedule_data.get("intervals", [])
    if not isinstance(intervals, list):
        raise ValueError("refer_schedule must contain an 'intervals' list")

    schedule_dir = os.path.dirname(schedule_src)
    interval_records = []
    for idx, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise ValueError(f"Interval entry at index {idx} must be a JSON object")
        path_value = interval.get("path")
        if not path_value:
            raise ValueError(f"Interval entry at index {idx} is missing the 'path' field")
        resolved_path = _resolve_reference_path(path_value, schedule_dir, save_path)
        interval_records.append(
            {
                "index": idx,
                "original_path": path_value,
                "resolved_path": resolved_path,
            }
        )

    return schedule_data, interval_records


if __name__ == '__main__':
    args = _parse_args()
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    assert not args.use_flux or args.retarget_flag, "Image editing with FLUX can only be used when pose retargeting is enabled."

    schedule_data = None
    schedule_records = []
    if args.refer_schedule is not None:
        if not os.path.isfile(args.refer_schedule):
            raise FileNotFoundError(f"refer_schedule file not found: {args.refer_schedule}")
        schedule_data, schedule_records = _load_reference_schedule(args.refer_schedule, args.save_path)

    front_source = args.refer_front_path or args.refer_path
    if front_source is None:
        raise ValueError("Please provide --refer_front_path (preferred) or --refer_path as the primary reference image.")

    front_path = _normalize_reference_file(front_source, "Front reference image")
    side_path = _normalize_reference_file(args.refer_side_path, "Side reference image") or front_path
    back_path = _normalize_reference_file(args.refer_back_path, "Back reference image") or front_path

    orientation_paths = {
        "front": front_path,
        "side": side_path,
        "back": back_path,
    }

    reference_candidates = [front_path, side_path, back_path]

    reference_candidates.extend(record["resolved_path"] for record in schedule_records)

    deduped_references = []
    seen_refs = set()
    for path in reference_candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen_refs:
            continue
        seen_refs.add(abs_path)
        deduped_references.append(abs_path)

    if not deduped_references:
        raise ValueError("At least one reference image must be provided via --refer_path or refer_schedule intervals.")

    pose2d_checkpoint_path = os.path.join(args.ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
    det_checkpoint_path = os.path.join(args.ckpt_path, 'det/yolov10m.onnx')

    sam2_checkpoint_path = os.path.join(args.ckpt_path, 'sam2/sam2_hiera_large.pt') if args.replace_flag else None
    flux_kontext_path = os.path.join(args.ckpt_path, 'FLUX.1-Kontext-dev') if args.use_flux else None
    process_pipeline = ProcessPipeline(det_checkpoint_path=det_checkpoint_path, pose2d_checkpoint_path=pose2d_checkpoint_path, sam_checkpoint_path=sam2_checkpoint_path, flux_kontext_path=flux_kontext_path)
    os.makedirs(args.save_path, exist_ok=True)
    pipeline_result = process_pipeline(video_path=args.video_path, 
                     refer_image_path=orientation_paths["front"],
                     refer_image_paths=deduped_references,
                     output_path=args.save_path,
                     resolution_area=args.resolution_area,
                     fps=args.fps,
                     iterations=args.iterations,
                     k=args.k,
                     w_len=args.w_len,
                     h_len=args.h_len,
                     retarget_flag=args.retarget_flag,
                     use_flux=args.use_flux,
                     replace_flag=args.replace_flag)

    orientation_filenames = {
        "front": "src_ref_front.png",
        "side": "src_ref_side.png",
        "back": "src_ref_back.png",
    }
    for orientation, src_path in orientation_paths.items():
        dst_path = os.path.join(args.save_path, orientation_filenames[orientation])
        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            continue
        shutil.copy(src_path, dst_path)

    if pipeline_result and pipeline_result.get("orientation_track"):
        orientation_track = list(pipeline_result.get("orientation_track", []))
        frame_count = pipeline_result.get("frame_count", len(orientation_track))
        orientation_meta = {
            "fps": pipeline_result.get("fps", args.fps),
            "frame_count": frame_count,
            "orientations": orientation_track,
            "labels": orientation_filenames,
        }
        orientation_path = os.path.join(args.save_path, 'src_orientation_track.json')
        with open(orientation_path, 'w', encoding='utf-8') as orientation_file:
            json.dump(orientation_meta, orientation_file, indent=2)
        print(f"Orientation track saved to {orientation_path}")

    if schedule_data is not None:
        ref_mapping = {}
        for idx, abs_path in enumerate(deduped_references):
            dst_name = 'src_ref.png' if idx == 0 else f'src_ref_{idx}.png'
            ref_mapping[abs_path] = dst_name

        for record in schedule_records:
            resolved_path = os.path.abspath(record["resolved_path"])
            if resolved_path not in ref_mapping:
                raise ValueError(f"Schedule reference {resolved_path} was not processed")
            schedule_data["intervals"][record["index"]]["path"] = ref_mapping[resolved_path]

        schedule_dst = os.path.join(args.save_path, 'src_ref_schedule.json')
        with open(schedule_dst, 'w', encoding='utf-8') as schedule_file:
            json.dump(schedule_data, schedule_file, indent=2)
        print(f"Reference schedule saved to {schedule_dst}")

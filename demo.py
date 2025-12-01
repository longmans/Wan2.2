import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent

# Default paths following run_animate.sh
PREPROCESS_SCRIPT = PROJECT_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = PROJECT_ROOT / "generate.py"
CKPT_PREPROCESS = "./Wan2.2-Animate-14B/process_checkpoint"
CKPT_GENERATE = "./Wan2.2-Animate-14B/"

# Where to store per-run intermediate data and outputs
RUN_ROOT = PROJECT_ROOT / "examples" / "wan_animate" / "gradio_runs"

# All selectable resolutions
RESOLUTION_OPTIONS = [
    "720*1280",   # default
    "1280*720",
    "480*832",
    "832*480",
    "704*1280",
    "1280*704",
    "1024*704",
    "704*1024",
]
DEFAULT_RESOLUTION = "720*1280"


def _parse_resolution(res_str: str) -> Tuple[int, int]:
    try:
        w_str, h_str = res_str.split("*")
        return int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError(f"Invalid resolution format: {res_str}") from exc


def _choose_base_size_for_animate(res_str: str) -> str:
    """
    animate-14B 在 generate.py 里只显式支持 720*1280 / 1280*720。
    这里根据长宽比自动映射到其中一个，用于 --size 参数。
    """
    w, h = _parse_resolution(res_str)
    if h >= w:
        return "720*1280"
    return "1280*720"


def _resize_video_if_needed(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
) -> str:
    """
    如果输入视频分辨率与目标不一致，则使用 OpenCV 重新编码到目标分辨率。
    返回最终可用的视频路径。
    """
    import cv2  # lazy import

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for resizing: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 已经是目标尺寸，直接返回原视频
    if src_w == target_w and src_h == target_h:
        cap.release()
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()
    return output_path


def _build_ref_schedule(
    ref_image_paths: List[str],
    intervals_table,
    save_path: Path,
) -> Optional[Path]:
    """
    根据用户在 Dataframe 中填写的 (image_index, start_sec, end_sec) 生成
    ref_schedule.json，并将参考图像拷贝到 save_path。

    返回 schedule 文件路径；如果没有有效区间则返回 None。
    """
    if not ref_image_paths:
        return None

    os.makedirs(save_path, exist_ok=True)

    intervals = []
    if intervals_table is not None:
        for row in intervals_table:
            if not row:
                continue
            if all(v in (None, "") for v in row):
                continue
            try:
                img_idx = int(row[0])
                start_sec = float(row[1])
                end_sec = float(row[2])
            except Exception:
                continue
            if img_idx < 0 or img_idx >= len(ref_image_paths):
                continue
            if end_sec <= start_sec:
                continue

            src_path = ref_image_paths[img_idx]
            base, ext = os.path.splitext(os.path.basename(src_path))
            if not ext:
                ext = ".png"
            dst_name = f"ref_{img_idx}{ext}"
            dst_path = save_path / dst_name
            if not dst_path.exists():
                shutil.copy(src_path, dst_path)

            intervals.append(
                {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "path": dst_name,
                }
            )

    if not intervals:
        # 没有有效区间就不创建 schedule，后续回退到单一参考图模式
        return None

    schedule = {"intervals": intervals}
    schedule_path = save_path / "ref_schedule.json"
    with schedule_path.open("w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)

    return schedule_path


def _run_subprocess(cmd: List[str], cwd: Path, log_prefix: str = ""):
    """
    以增量方式读取子进程 stdout，用于在 Gradio 中实时展示日志。
    这是一个生成器：yield 每次更新后的日志字符串。
    """
    full_cmd_str = " ".join(cmd)
    logs = f"{log_prefix}$ {full_cmd_str}\n"
    yield logs

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logs += line
        yield logs

    process.wait()
    if process.returncode != 0:
        logs += f"\n[ERROR] Command failed with return code {process.returncode}\n"
        yield logs
        raise RuntimeError(f"Command failed: {full_cmd_str}")

    yield logs


def run_animate_pipeline(
    driving_video: str,
    ref_images: List[str],
    intervals_table,
    resolution: str,
):
    """
    Gradio 回调：执行预处理 + 生成流程，并在页面中展示日志和结果视频。
    """
    if not driving_video:
        return "请先上传待复刻视频。", None
    if not ref_images:
        return "请至少上传一张参考图片。", None

    # 解析分辨率与 animate-14B 的基础 size
    try:
        target_w, target_h = _parse_resolution(resolution)
    except Exception as exc:
        return f"分辨率解析失败: {exc}", None
    base_size_for_generate = _choose_base_size_for_animate(resolution)

    # 为本次运行创建独立目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # 复制 driving video 以便归档（pipeline 可以继续直接用上传临时路径）
    video_ext = os.path.splitext(driving_video)[1] or ".mp4"
    archived_video_path = run_dir / f"driving{video_ext}"
    try:
        shutil.copy(driving_video, archived_video_path)
    except Exception:
        # 复制失败不影响主流程
        pass

    # 构建参考 schedule（如果用户填写了区间）
    schedule_path = _build_ref_schedule(ref_images, intervals_table, run_dir)

    # 日志初始化
    all_logs = ""
    all_logs += "=== Wan2.2 Animate Demo ===\n"
    all_logs += f"Run directory: {run_dir}\n"
    all_logs += f"Driving video: {driving_video}\n"
    all_logs += "Reference images:\n"
    for idx, p in enumerate(ref_images):
        all_logs += f"  [{idx}] {p}\n"
    all_logs += f"Target resolution: {resolution}\n"
    all_logs += f"Generate base size (animate-14B): {base_size_for_generate}\n"
    if schedule_path is not None:
        all_logs += f"Using reference schedule: {schedule_path}\n"
    else:
        all_logs += "No valid schedule provided, fallback to single reference mode.\n"

    yield all_logs, None

    # 1) 预处理
    preprocess_cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--ckpt_path",
        CKPT_PREPROCESS,
        "--video_path",
        driving_video,
        "--save_path",
        str(run_dir),
        "--resolution_area",
        str(target_w),
        str(target_h),
        "--retarget_flag",
        "--use_flux",
    ]

    if schedule_path is not None:
        preprocess_cmd += ["--refer_schedule", str(schedule_path)]
    else:
        # 单一参考图模式，使用第一张图
        preprocess_cmd += ["--refer_path", ref_images[0]]

    try:
        for logs in _run_subprocess(preprocess_cmd, PROJECT_ROOT, log_prefix="[preprocess] "):
            merged = all_logs + logs
            yield merged, None
        all_logs += logs
    except Exception as exc:
        all_logs += f"\n预处理阶段失败: {exc}\n"
        yield all_logs, None
        return

    # 2) 生成视频
    base_output_video = run_dir / "output_base.mp4"
    generate_cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--task",
        "animate-14B",
        "--ckpt_dir",
        CKPT_GENERATE,
        "--src_root_path",
        str(run_dir),
        "--refert_num",
        "5",
        "--size",
        base_size_for_generate,
        "--save_file",
        str(base_output_video),
    ]

    try:
        for logs in _run_subprocess(generate_cmd, PROJECT_ROOT, log_prefix="[generate] "):
            merged = all_logs + logs
            yield merged, None
        all_logs += logs
    except Exception as exc:
        all_logs += f"\n生成阶段失败: {exc}\n"
        yield all_logs, None
        return

    # 3) 如有需要，对生成视频进行二次缩放到用户选择的分辨率
    final_video_path = run_dir / "output.mp4"
    try:
        final_video = _resize_video_if_needed(
            input_path=str(base_output_video),
            output_path=str(final_video_path),
            target_w=target_w,
            target_h=target_h,
        )
    except Exception as exc:
        all_logs += f"\n[WARN] 调整输出分辨率失败，将直接使用原始生成视频: {exc}\n"
        final_video = str(base_output_video)

    all_logs += f"\n生成完成，最终视频路径: {final_video}\n"
    yield all_logs, final_video


def build_demo():
    with gr.Blocks(title="Wan2.2 Animate Demo") as demo:
        gr.Markdown(
            """
## Wan2.2 Animate Demo

上传待复刻视频和一张或多张参考图片，可为每张参考图片设置多个时间区间。

- `image_index` 为参考图片在上传顺序中的索引（从 0 开始）
- `start_sec` / `end_sec` 为该参考图生效的起止时间（单位：秒），区间为左闭右开 `[start_sec, end_sec)`，即不包含 `end_sec` 那一刻
"""
        )

        with gr.Row():
            driving_video = gr.File(
                label="待复刻视频",
                file_types=["video"],
                type="filepath",
            )
            ref_images = gr.File(
                label="参考图片（可多张）",
                file_types=["image"],
                file_count="multiple",
                type="filepath",
            )

        intervals = gr.Dataframe(
            headers=["image_index", "start_sec", "end_sec"],
            datatype=["number", "number", "number"],
            row_count=(1, "dynamic"),
            col_count=3,
            label="参考图片时间区间（可留空，仅使用首张参考图）",
        )

        resolution = gr.Dropdown(
            choices=RESOLUTION_OPTIONS,
            value=DEFAULT_RESOLUTION,
            label="输出分辨率（宽*高）",
        )

        run_button = gr.Button("生成", variant="primary")

        logs = gr.Textbox(
            label="运行日志",
            lines=20,
            interactive=False,
        )
        output_video = gr.Video(
            label="生成结果视频",
        )

        run_button.click(
            fn=run_animate_pipeline,
            inputs=[driving_video, ref_images, intervals, resolution],
            outputs=[logs, output_video],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    os.makedirs(RUN_ROOT, exist_ok=True)
    demo = build_demo()
    demo.queue().launch(share=True)

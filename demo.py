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
]
DEFAULT_RESOLUTION = "720*1280"


def _parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse strings like '720*1280' into (width, height)."""
    try:
        w_str, h_str = res_str.split("*")
        return int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError(f"Invalid resolution format: {res_str}") from exc


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

    table_rows = []
    if intervals_table is None:
        table_rows = []
    elif hasattr(intervals_table, "values"):
        table_rows = intervals_table.values.tolist()
    else:
        table_rows = intervals_table

    intervals = []
    if table_rows:
        for row in table_rows:
            if not row:
                continue
            try:
                img_idx = int(row[0])
            except Exception:
                continue
            if img_idx < 0 or img_idx >= len(ref_image_paths):
                continue

            interval_text = row[1] if len(row) > 1 else ""
            parsed_ranges = _parse_interval_ranges(interval_text)
            if not parsed_ranges:
                continue

            src_path = ref_image_paths[img_idx]
            base, ext = os.path.splitext(os.path.basename(src_path))
            if not ext:
                ext = ".png"
            dst_name = f"ref_{img_idx}{ext}"
            dst_path = save_path / dst_name
            if not dst_path.exists():
                shutil.copy(src_path, dst_path)

            for start_sec, end_sec in parsed_ranges:
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


def _summarize_ref_images(ref_image_paths: Optional[List[str]]):
    """Return gallery data and markdown text for uploaded reference images."""
    if not ref_image_paths:
        return [], "尚未上传参考图片，上传后会显示索引。"

    gallery_items = [
        (path, f"[{idx}] {os.path.basename(path)}")
        for idx, path in enumerate(ref_image_paths)
    ]
    text_lines = ["**参考图片索引对照**", "按照上传顺序自动编号："]
    for idx, path in enumerate(ref_image_paths):
        text_lines.append(f"- [{idx}] {os.path.basename(path)}")
    return gallery_items, "\n".join(text_lines)


def _prepare_ref_image_ui(ref_image_paths: Optional[List[str]], current_table=None):
    """Sync gallery, markdown, and interval rows when reference images change."""
    gallery_items, hint_text = _summarize_ref_images(ref_image_paths)

    interval_rows: List[List[object]] = []
    preserved = {}
    table_rows = []
    if current_table is None:
        table_rows = []
    elif hasattr(current_table, "values"):
        table_rows = current_table.values.tolist()
    else:
        table_rows = current_table

    if table_rows:
        for row in table_rows:
            if not row:
                continue
            try:
                idx = int(row[0])
            except (TypeError, ValueError):
                continue
            preserved[idx] = row[1] if len(row) > 1 and row[1] is not None else ""

    if ref_image_paths:
        for idx in range(len(ref_image_paths)):
            interval_rows.append([idx, preserved.get(idx, "")])

    return gallery_items, hint_text, gr.update(value=interval_rows)


def _parse_interval_ranges(interval_text: Optional[str]) -> List[Tuple[float, float]]:
    """Parse comma-separated `start-end` segments into float tuples."""
    if not interval_text:
        return []

    normalized = interval_text.replace("，", ",")
    ranges: List[Tuple[float, float]] = []
    for token in normalized.split(","):
        token = token.strip()
        if not token or "-" not in token:
            continue
        start_str, end_str = token.split("-", 1)
        try:
            start = float(start_str.strip())
            end = float(end_str.strip())
        except ValueError:
            continue
        if start == end:
            continue
        if end < start:
            start, end = end, start
        ranges.append((start, end))

    return ranges


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
    front_ref_index: Optional[float],
):
    """
    Gradio 回调：执行预处理 + 生成流程，并在页面中展示日志和结果视频。
    """
    if not driving_video:
        return "请先上传待复刻视频。", None
    if not ref_images:
        return "请至少上传一张参考图片。", None

    # 选择 front_refer_path 对应的参考图索引（单选）
    try:
        if front_ref_index is None:
            front_idx = 0
        else:
            front_idx = int(front_ref_index)
    except Exception:
        front_idx = 0
    if front_idx < 0 or front_idx >= len(ref_images):
        front_idx = 0
    front_ref_path = ref_images[front_idx]

    try:
        target_w, target_h = _parse_resolution(resolution)
    except Exception as exc:
        return f"分辨率解析失败: {exc}", None

    base_size_for_generate = resolution

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
    all_logs += f"Front reference index (front_refer_path): {front_idx}\n"
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

    # 始终把选定的 front_refer_path 传给预处理脚本，用于生成固定正面参考图
    if front_ref_path:
        preprocess_cmd += ["--front_refer_path", front_ref_path]

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
        "--offload_model", 
        "False"
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

    # 3) 直接返回生成脚本输出的视频
    final_video = str(base_output_video)
    all_logs += f"\n生成完成，最终视频路径: {final_video}\n"
    yield all_logs, final_video


def build_demo():
    with gr.Blocks(title="Wan2.2 Animate Demo") as demo:
        gr.Markdown(
            """
## Wan2.2 Animate Demo

上传待复刻视频和一张或多张参考图片，可为每张参考图片设置多个时间区间。

- 每张参考图在下方“参考图片时间区间”表格中会自动生成一行，左侧展示索引，右侧可填写形如 `0-2,5-8` 的区间列表（多个区间用逗号隔开，可填写整数或小数）。
- 区间为左闭右开 `[start_sec, end_sec)`，也就是包含开始的数值，不包含结束的数值
"""
        )

        with gr.Row():
            driving_video = gr.Video(
                label="待复刻视频（上传后自动预览）",
                sources=["upload"],
                interactive=True,
            )
            ref_images = gr.File(
                label="参考图片（可多张）",
                file_types=["image"],
                file_count="multiple",
                type="filepath",
            )

        image_gallery = gr.Gallery(
            label="已上传参考图预览",
            value=[],
            columns=4,
            height="auto",
        )
        image_index_hint = gr.Markdown("上传参考图后，这里会展示索引对照表。")

        intervals = gr.Dataframe(
            headers=["image_index", "intervals"],
            datatype=["number", "str"],
            row_count=(0, "dynamic"),
            col_count=2,
            label="参考图片时间区间（右侧输入 0-2,5-8 形式的区间，逗号分隔, 区间为左闭右开 `[start_sec, end_sec)`，也就是包含开始的数值，不包含结束的数值）",
            type="array",
        )

        resolution = gr.Dropdown(
            choices=RESOLUTION_OPTIONS,
            value=DEFAULT_RESOLUTION,
            label="输出分辨率（宽*高）",
        )

        front_ref_index = gr.Number(
            label="正面参考图索引（front_refer_path，对应上方索引，整数）",
            value=0,
            precision=0,
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
            inputs=[driving_video, ref_images, intervals, resolution, front_ref_index],
            outputs=[logs, output_video],
            queue=True,
        )

        ref_images.change(
            fn=_prepare_ref_image_ui,
            inputs=[ref_images, intervals],
            outputs=[image_gallery, image_index_hint, intervals],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    os.makedirs(RUN_ROOT, exist_ok=True)
    demo = build_demo()
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=8888)

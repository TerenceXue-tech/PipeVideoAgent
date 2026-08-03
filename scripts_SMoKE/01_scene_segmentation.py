# -*- coding: utf-8 -*-
"""
使用 SigLIP2 提取视频帧 embedding，并计算固定步长下的帧间距离曲线。

输出：
1. 每个视频的步长帧余弦相似度曲线图
2. 每个视频的步长帧变化距离曲线图：change_distance = 1 - cosine_similarity
3. 每个视频的灰度直方图 Bhattacharyya 距离曲线图
4. 每个视频的 RGB 直方图 Bhattacharyya 距离曲线图
5. 每个视频的 embedding.npy
6. 每个视频的各方法 scores.csv
7. 每个视频的 sampled_frames 元数据与候选边界（按灰度直方图结果）

建议用途：
- cosine_similarity 越低，说明步长帧差异越大；
- change_distance 越高，说明越可能是场景/视角切换点；
- 场景分割候选边界默认采用灰度直方图 Bhattacharyya 距离峰值。
"""

import os
import csv
import math
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel


# =========================
# 1. 固定参数区域
# =========================

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/xtc/LLMs/siglip2-giant-opt-patch16-384")
VIDEO_LIST_PATH = os.environ.get("VIDEO_LIST_PATH", "/home/xtc/PipeVideo/video_list.json")
with open(VIDEO_LIST_PATH, "r", encoding="utf-8") as f:
    VIDEO_LIST = json.load(f)
if not isinstance(VIDEO_LIST, list):
    raise ValueError(f"VIDEO_LIST 必须是 list，当前类型: {type(VIDEO_LIST)}")
for idx, item in enumerate(VIDEO_LIST):
    if not isinstance(item, dict) or "video_path" not in item:
        raise ValueError(f"VIDEO_LIST[{idx}] 格式错误，必须包含 video_path 字段，当前值: {item}")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/xtc/PipeVideo/scene_segmentation_outputs")

# 抽帧频率：每秒抽几帧
# 如果视频较长，建议 0.5 或 1.0；
# 如果想看更细的变化，可以设为 4.0。
SAMPLE_FPS = 4

# 帧间比较步长（基于采样后的帧索引）
# 默认等于 SAMPLE_FPS，即大约比较“相隔 1 秒”的两帧。
# 设为 None 表示使用 SAMPLE_FPS；若显式设置为整数 N，则比较 frame_i 与 frame_{i+N}。
# 示例：SAMPLE_FPS=4 且 COSINE_COMPARE_STEP=4，则比较 frame_i 与 frame_{i+4}。
COSINE_COMPARE_STEP = None

# SigLIP2 giant 显存占用
BATCH_SIZE = 256

# 是否保存抽出的帧图片
SAVE_SAMPLED_FRAMES = False

# 是否使用 float16 推理
USE_FP16 = True

# 余弦变化距离阈值：mean + THRESHOLD_STD_FACTOR * std
# 仅用于图中标出候选切换点，不影响 embedding 和相似度计算。
THRESHOLD_STD_FACTOR = 2.5

# 候选点之间的最小时间间隔，单位秒（5 秒内不重复）
MIN_PEAK_GAP_SECONDS = 5.0

# 预置位切换过程可视化窗口改为自适应：
# 对每个候选边界点，分别向前/向后找到“最近的低于均值点”，
# 并将这两个点构成的时间区间涂灰。

# 灰度 / RGB 直方图 bins 数（每通道），对比公平
GRAY_HIST_BINS = 1536
RGB_HIST_BINS = 512

# 场景分割候选边界来源：按灰度直方图距离做最终分割
BOUNDARY_SOURCE = "gray_hist_bhattacharyya"


# =========================
# 2. 工具函数
# =========================

def safe_video_name(video_path: str) -> str:
    """
    从视频路径生成安全的输出目录名。
    """
    p = Path(video_path)
    return p.stem.replace(" ", "_")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_siglip2_model(model_path: str, device: torch.device):
    """
    加载 SigLIP2 processor 和 model。
    """
    print(f"[LOAD] processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"[LOAD] model: {model_path}")
    if device.type == "cuda" and USE_FP16:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
        )

    model.to(device)
    model.eval()

    return processor, model


def sample_video_frames(
    video_path: str,
    sample_fps: float,
    save_dir: str = None,
) -> Tuple[List[Image.Image], List[Dict]]:
    """
    按 sample_fps 从视频中均匀抽帧。

    返回：
    frames: PIL.Image 列表
    metas: 每帧元信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0

    if native_fps <= 0:
        raise RuntimeError(f"无法读取视频 FPS: {video_path}")

    sample_interval = max(1, int(round(native_fps / sample_fps)))

    print(f"[VIDEO] {video_path}")
    print(f"        native_fps={native_fps:.3f}, total_frames={total_frames}, duration={duration:.2f}s")
    print(f"        sample_fps={sample_fps}, sample_interval={sample_interval} frames")

    if save_dir is not None:
        ensure_dir(save_dir)

    frames: List[Image.Image] = []
    metas: List[Dict] = []

    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp_sec = frame_idx / native_fps

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            frames.append(pil_img)

            frame_name = f"frame_{sampled_idx:06d}_t{timestamp_sec:.2f}.jpg"
            frame_save_path = ""

            if save_dir is not None:
                frame_save_path = os.path.join(save_dir, frame_name)
                pil_img.save(frame_save_path, quality=95)

            metas.append(
                {
                    "sampled_index": sampled_idx,
                    "original_frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "frame_save_path": frame_save_path,
                }
            )

            sampled_idx += 1

        frame_idx += 1

    cap.release()

    print(f"[SAMPLE] sampled_frames={len(frames)}")

    if len(frames) < 2:
        raise RuntimeError(f"抽帧数量少于 2，无法计算相邻帧相似度: {video_path}")

    return frames, metas


@torch.inference_mode()
def extract_image_embeddings(
    frames: List[Image.Image],
    processor,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    提取每帧全局图像 embedding。

    返回：
    embeddings: [T, D] numpy array
    """

    def pick_feature_tensor(outputs):
        """
        从不同类型的模型输出中提取真正的图像特征 Tensor。
        兼容：
        1. Tensor
        2. BaseModelOutputWithPooling
        3. 带 image_embeds / pooler_output / last_hidden_state 的对象
        """
        if torch.is_tensor(outputs):
            return outputs

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            # patch token 平均池化成全局图像向量
            return outputs.last_hidden_state.mean(dim=1)

        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if torch.is_tensor(item):
                    return item

        raise RuntimeError(f"无法从模型输出中提取特征，输出类型为: {type(outputs)}")

    all_embeddings = []

    num_frames = len(frames)
    num_batches = math.ceil(num_frames / batch_size)

    for batch_id in range(num_batches):
        start = batch_id * batch_size
        end = min(start + batch_size, num_frames)
        batch_images = frames[start:end]

        inputs = processor(
            images=batch_images,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if torch.is_tensor(v)
        }

        # 优先走 get_image_features，但要兼容其返回对象而不是 Tensor 的情况
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
            image_features = pick_feature_tensor(outputs)
        else:
            # 如果没有 get_image_features，就直接走 vision_model
            if hasattr(model, "vision_model"):
                outputs = model.vision_model(**inputs)
            else:
                outputs = model(**inputs)

            image_features = pick_feature_tensor(outputs)

        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=-1)

        all_embeddings.append(image_features.cpu())

        print(
            f"[EMBED] batch {batch_id + 1}/{num_batches}, "
            f"frames {start}-{end - 1}, "
            f"feature_shape={tuple(image_features.shape)}"
        )

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    print(f"[EMBED] embeddings.shape = {embeddings.shape}")

    return embeddings


def compute_adjacent_cosine_scores(
    embeddings: np.ndarray,
    metas: List[Dict],
    pair_step: int,
) -> List[Dict]:
    """
    计算固定步长帧余弦相似度和变化距离。

    embeddings 已经归一化时：
    cosine_similarity = dot(e_i, e_{i+pair_step})
    change_distance = 1 - cosine_similarity
    """
    if pair_step <= 0:
        raise ValueError(f"pair_step 必须 >= 1，当前为: {pair_step}")
    if len(metas) <= pair_step:
        raise ValueError(
            f"采样帧数量不足以按步长计算距离：sampled_frames={len(metas)}, pair_step={pair_step}"
        )

    emb = torch.from_numpy(embeddings).float()
    emb = F.normalize(emb, dim=1)

    cosine_sim = (emb[:-pair_step] * emb[pair_step:]).sum(dim=1).cpu().numpy()
    change_distance = 1.0 - cosine_sim

    rows = []

    for i in range(len(cosine_sim)):
        t1 = metas[i]["timestamp_sec"]
        t2 = metas[i + pair_step]["timestamp_sec"]
        t_mid = (t1 + t2) / 2.0

        rows.append(
            {
                "pair_index": i,
                "pair_step": pair_step,
                "frame_i": metas[i]["sampled_index"],
                "frame_j": metas[i + pair_step]["sampled_index"],
                "time_i_sec": t1,
                "time_j_sec": t2,
                "time_mid_sec": t_mid,
                "cosine_similarity": float(cosine_sim[i]),
                "change_distance": float(change_distance[i]),
                "frame_i_path": metas[i].get("frame_save_path", ""),
                "frame_j_path": metas[i + pair_step].get("frame_save_path", ""),
            }
        )

    return rows


def grayscale_histogram(image_np: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("GRAY_HIST_BINS 必须大于 0")

    if image_np.ndim == 3:
        arr = image_np.astype(np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = image_np.astype(np.float32)

    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 256.0))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return np.full((bins,), 1.0 / bins, dtype=np.float64)
    return hist / total


def rgb_histogram(image_np: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("RGB_HIST_BINS 必须大于 0")

    if image_np.ndim != 3 or image_np.shape[2] < 3:
        return grayscale_histogram(image_np, bins=bins)

    arr = image_np[:, :, :3].astype(np.float32)
    hists: List[np.ndarray] = []

    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0.0, 256.0))
        hist = hist.astype(np.float64)
        total = float(hist.sum())
        if total <= 0:
            hist = np.full((bins,), 1.0 / bins, dtype=np.float64)
        else:
            hist = hist / total
        hists.append(hist)

    merged = np.concatenate(hists, axis=0)
    total = float(merged.sum())
    if total <= 0:
        return np.full((bins * 3,), 1.0 / (bins * 3), dtype=np.float64)
    return merged / total


def bhattacharyya_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    coeff = float(np.sum(np.sqrt(hist_a * hist_b)))
    coeff = max(coeff, 1e-12)
    return float(-math.log(coeff))


def compute_histogram_bhattacharyya_scores(
    frames: List[Image.Image],
    metas: List[Dict],
    pair_step: int,
    hist_method: str,
    bins: int,
) -> List[Dict]:
    if pair_step <= 0:
        raise ValueError(f"pair_step 必须 >= 1，当前为: {pair_step}")
    if len(metas) <= pair_step:
        raise ValueError(
            f"采样帧数量不足以按步长计算距离：sampled_frames={len(metas)}, pair_step={pair_step}"
        )

    hist_method = hist_method.lower().strip()
    if hist_method == "gray":
        hist_fn = grayscale_histogram
    elif hist_method == "rgb":
        hist_fn = rgb_histogram
    else:
        raise ValueError(f"不支持的 hist_method: {hist_method}")

    hist_list: List[np.ndarray] = []
    for frame in frames:
        frame_np = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        hist_list.append(hist_fn(frame_np, bins=bins))

    rows = []
    for i in range(len(hist_list) - pair_step):
        t1 = metas[i]["timestamp_sec"]
        t2 = metas[i + pair_step]["timestamp_sec"]
        t_mid = (t1 + t2) / 2.0
        distance = bhattacharyya_distance(hist_list[i], hist_list[i + pair_step])

        rows.append(
            {
                "pair_index": i,
                "pair_step": pair_step,
                "frame_i": metas[i]["sampled_index"],
                "frame_j": metas[i + pair_step]["sampled_index"],
                "time_i_sec": t1,
                "time_j_sec": t2,
                "time_mid_sec": t_mid,
                "hist_method": hist_method,
                "hist_bins": bins,
                "bhattacharyya_distance": float(distance),
                "frame_i_path": metas[i].get("frame_save_path", ""),
                "frame_j_path": metas[i + pair_step].get("frame_save_path", ""),
            }
        )

    return rows


def find_candidate_peaks(
    rows: List[Dict],
    threshold_std_factor: float,
    min_gap_seconds: float,
    distance_key: str = "change_distance",
) -> Tuple[float, List[int]]:
    """
    在指定距离曲线上找候选点：
    1) 必须超过阈值；
    2) 按时间顺序取最早出现点；
    3) 5 秒（或 min_gap_seconds）内不重复。
    返回：
    threshold
    peak_indices: rows 里的 pair_index 列表
    """
    if not rows:
        return 0.0, []
    if distance_key not in rows[0]:
        raise KeyError(f"rows 中不存在 distance_key={distance_key}")

    distances = np.array([r[distance_key] for r in rows], dtype=np.float32)
    times = np.array([r["time_mid_sec"] for r in rows], dtype=np.float32)

    mean = float(distances.mean())
    std = float(distances.std())
    threshold = mean + threshold_std_factor * std

    selected = []
    last_selected_time = -1e12

    for idx in range(len(distances)):
        t = float(times[idx])
        is_over_threshold = float(distances[idx]) > threshold
        is_gap_ok = (t - last_selected_time) >= min_gap_seconds
        if is_over_threshold and is_gap_ok:
            selected.append(idx)
            last_selected_time = t

    return threshold, selected


def compute_adaptive_windows(
    rows: List[Dict],
    peak_indices: List[int],
    value_key: str,
) -> List[Dict]:
    """
    基于均值的自适应窗口：
    对每个边界点，向前/向后找最近低于均值的点。
    """
    if not rows or not peak_indices:
        return []
    if value_key not in rows[0]:
        raise KeyError(f"rows 中不存在 value_key={value_key}")

    times = np.array([float(r["time_mid_sec"]) for r in rows], dtype=np.float32)
    values = np.array([float(r[value_key]) for r in rows], dtype=np.float32)
    mean_value = float(values.mean())

    windows: List[Dict] = []
    for idx in peak_indices:
        if idx < 0 or idx >= len(rows):
            continue

        left_idx = 0
        for j in range(idx - 1, -1, -1):
            if values[j] < mean_value:
                left_idx = j
                break

        right_idx = len(rows) - 1
        for j in range(idx + 1, len(rows)):
            if values[j] < mean_value:
                right_idx = j
                break

        t_center = float(times[idx])
        t_left = float(times[left_idx])
        t_right = float(times[right_idx])

        windows.append(
            {
                "boundary_pair_index": int(rows[idx]["pair_index"]),
                "boundary_time_sec": t_center,
                "mean_value": mean_value,
                "left_pair_index": int(rows[left_idx]["pair_index"]),
                "left_time_sec": t_left,
                "left_value": float(values[left_idx]),
                "right_pair_index": int(rows[right_idx]["pair_index"]),
                "right_time_sec": t_right,
                "right_value": float(values[right_idx]),
                "adaptive_pre_seconds": max(0.0, t_center - t_left),
                "adaptive_post_seconds": max(0.0, t_right - t_center),
            }
        )

    return windows


def save_scores_csv(rows: List[Dict], csv_path: str):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    rows: List[Dict],
    y_key: str,
    title: str,
    ylabel: str,
    save_path: str,
    threshold: float = None,
    peak_indices: List[int] = None,
    peak_rows: List[Dict] = None,
):
    times = np.array([r["time_mid_sec"] for r in rows], dtype=np.float32)
    values = np.array([r[y_key] for r in rows], dtype=np.float32)
    mean_value = float(values.mean()) if len(values) > 0 else 0.0

    plt.figure(figsize=(14, 5))

    peak_times = None
    peak_values = None

    if peak_rows:
        peak_times = np.array([float(r["time_mid_sec"]) for r in peak_rows], dtype=np.float32)
        # 通过时间戳在当前曲线上找最近点，避免跨方法索引错位
        nearest_indices = np.array(
            [int(np.argmin(np.abs(times - t))) for t in peak_times],
            dtype=np.int32,
        )
        peak_values = values[nearest_indices]
    elif peak_indices:
        peak_times = times[peak_indices]
        peak_values = values[peak_indices]

    adaptive_windows: List[Tuple[float, float]] = []
    adaptive_point_indices: List[int] = []
    if peak_times is not None and len(peak_times) > 0:
        for t in peak_times:
            center_idx = int(np.argmin(np.abs(times - float(t))))

            left_idx = 0
            for j in range(center_idx - 1, -1, -1):
                if values[j] < mean_value:
                    left_idx = j
                    break

            right_idx = len(values) - 1
            for j in range(center_idx + 1, len(values)):
                if values[j] < mean_value:
                    right_idx = j
                    break

            adaptive_windows.append((float(times[left_idx]), float(times[right_idx])))
            adaptive_point_indices.append(left_idx)
            adaptive_point_indices.append(right_idx)

    if adaptive_windows:
        for idx, (left_t, right_t) in enumerate(adaptive_windows):
            label = "adaptive switch window" if idx == 0 else None
            plt.axvspan(
                left_t,
                right_t,
                color="gray",
                alpha=0.20,
                linewidth=0,
                label=label,
                zorder=0,
            )

    plt.plot(times, values, linewidth=1.5, zorder=2)
    plt.axhline(
        mean_value,
        color="blue",
        linestyle="--",
        linewidth=1.2,
        label=f"mean={mean_value:.4f}",
    )

    if threshold is not None:
        plt.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"threshold={threshold:.4f}",
        )

    if peak_times is not None and len(peak_times) > 0:
        plt.scatter(
            peak_times,
            peak_values,
            s=35,
            color="red",
            label="candidate boundaries",
            zorder=3,
        )
    if adaptive_point_indices:
        unique_idx = sorted(set(int(i) for i in adaptive_point_indices))
        adaptive_times = times[unique_idx]
        adaptive_values = values[unique_idx]
        plt.scatter(
            adaptive_times,
            adaptive_values,
            s=30,
            color="blue",
            label="adaptive low points",
            zorder=3,
        )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if threshold is not None or peak_indices:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# 3. 主流程
# =========================

def process_one_video(
    video_item: Dict,
    processor,
    model,
    device: torch.device,
):
    video_path = video_item["video_path"]
    video_name = safe_video_name(video_path)

    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    ensure_dir(video_out_dir)

    frames_dir = os.path.join(video_out_dir, "sampled_frames") if SAVE_SAMPLED_FRAMES else None
    if not SAVE_SAMPLED_FRAMES:
        stale_frames_dir = os.path.join(video_out_dir, "sampled_frames")
        if os.path.isdir(stale_frames_dir):
            shutil.rmtree(stale_frames_dir)
            print(f"[CLEAN] removed stale sampled frames dir: {stale_frames_dir}")

    print("\n" + "=" * 100)
    print(f"[START] {video_path}")
    print(f"[OUT]   {video_out_dir}")

    step_source = SAMPLE_FPS if COSINE_COMPARE_STEP is None else COSINE_COMPARE_STEP
    pair_step = max(1, int(round(float(step_source))))
    print(f"[CONFIG] sample_fps={SAMPLE_FPS}, pair_step={pair_step}")

    frames, metas = sample_video_frames(
        video_path=video_path,
        sample_fps=SAMPLE_FPS,
        save_dir=frames_dir,
    )

    embeddings = extract_image_embeddings(
        frames=frames,
        processor=processor,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
    )

    rows = compute_adjacent_cosine_scores(
        embeddings=embeddings,
        metas=metas,
        pair_step=pair_step,
    )

    gray_rows = compute_histogram_bhattacharyya_scores(
        frames=frames,
        metas=metas,
        pair_step=pair_step,
        hist_method="gray",
        bins=GRAY_HIST_BINS,
    )

    rgb_rows = compute_histogram_bhattacharyya_scores(
        frames=frames,
        metas=metas,
        pair_step=pair_step,
        hist_method="rgb",
        bins=RGB_HIST_BINS,
    )

    threshold, peak_indices = find_candidate_peaks(
        rows=rows,
        threshold_std_factor=THRESHOLD_STD_FACTOR,
        min_gap_seconds=MIN_PEAK_GAP_SECONDS,
        distance_key="change_distance",
    )

    gray_threshold, gray_peak_indices = find_candidate_peaks(
        rows=gray_rows,
        threshold_std_factor=THRESHOLD_STD_FACTOR,
        min_gap_seconds=MIN_PEAK_GAP_SECONDS,
        distance_key="bhattacharyya_distance",
    )

    rgb_threshold, rgb_peak_indices = find_candidate_peaks(
        rows=rgb_rows,
        threshold_std_factor=THRESHOLD_STD_FACTOR,
        min_gap_seconds=MIN_PEAK_GAP_SECONDS,
        distance_key="bhattacharyya_distance",
    )

    # 给 rows 加上候选边界标记
    # 注意：scene segmentation 的最终边界来源采用灰度直方图。
    peak_set = set(peak_indices)
    for r in rows:
        r["is_candidate_boundary"] = int(r["pair_index"] in peak_set)
    gray_peak_set = set(gray_peak_indices)
    for r in gray_rows:
        r["is_candidate_boundary"] = int(r["pair_index"] in gray_peak_set)
    rgb_peak_set = set(rgb_peak_indices)
    for r in rgb_rows:
        r["is_candidate_boundary"] = int(r["pair_index"] in rgb_peak_set)

    # 保存 embedding
    embedding_path = os.path.join(video_out_dir, "frame_embeddings.npy")
    np.save(embedding_path, embeddings)

    # 保存帧元数据
    metas_path = os.path.join(video_out_dir, "sampled_frames_meta.json")
    save_json(metas, metas_path)

    # 保存分数 CSV
    csv_path = os.path.join(video_out_dir, "cosine_step_scores.csv")
    gray_csv_path = os.path.join(video_out_dir, "gray_hist_bhattacharyya_scores.csv")
    rgb_csv_path = os.path.join(video_out_dir, "rgb_hist_bhattacharyya_scores.csv")
    save_scores_csv(rows, csv_path)
    save_scores_csv(gray_rows, gray_csv_path)
    save_scores_csv(rgb_rows, rgb_csv_path)

    # 保存候选边界 JSON（按灰度直方图峰值）
    gray_adaptive_windows = compute_adaptive_windows(
        rows=gray_rows,
        peak_indices=gray_peak_indices,
        value_key="bhattacharyya_distance",
    )
    adaptive_map = {
        int(item["boundary_pair_index"]): item
        for item in gray_adaptive_windows
    }
    candidate_boundaries = []
    for i in gray_peak_indices:
        item = dict(gray_rows[i])
        pair_index = int(item["pair_index"])
        adaptive_item = adaptive_map.get(pair_index)
        if adaptive_item is not None:
            item.update(
                {
                    "adaptive_pre_seconds": float(adaptive_item["adaptive_pre_seconds"]),
                    "adaptive_post_seconds": float(adaptive_item["adaptive_post_seconds"]),
                    "adaptive_left_time_sec": float(adaptive_item["left_time_sec"]),
                    "adaptive_right_time_sec": float(adaptive_item["right_time_sec"]),
                }
            )
        candidate_boundaries.append(item)
    boundary_path = os.path.join(video_out_dir, "candidate_boundaries.json")
    save_json(
        {
            "boundary_source": BOUNDARY_SOURCE,
            "threshold": gray_threshold,
            "threshold_rule": f"mean + {THRESHOLD_STD_FACTOR} * std",
            "min_peak_gap_seconds": MIN_PEAK_GAP_SECONDS,
            "pair_step": pair_step,
            "cosine_threshold": threshold,
            "gray_threshold": gray_threshold,
            "rgb_threshold": rgb_threshold,
            "num_candidate_boundaries": len(candidate_boundaries),
            "candidate_boundaries": candidate_boundaries,
            "adaptive_windows_rule": "for each boundary, nearest points below mean on both sides",
            "adaptive_windows": gray_adaptive_windows,
            "num_gray_hist_candidate_boundaries": len(gray_peak_indices),
            "num_rgb_hist_candidate_boundaries": len(rgb_peak_indices),
        },
        boundary_path,
    )

    # 绘制余弦相似度曲线
    cosine_png = os.path.join(video_out_dir, "cosine_similarity_curve.png")
    plot_curve(
        rows=rows,
        y_key="cosine_similarity",
        title=f"Step-{pair_step} Frame Cosine Similarity - {video_name}",
        ylabel="Cosine Similarity",
        save_path=cosine_png,
        threshold=None,
        peak_indices=peak_indices,
    )

    # 绘制变化距离曲线
    distance_png = os.path.join(video_out_dir, "change_distance_curve.png")
    plot_curve(
        rows=rows,
        y_key="change_distance",
        title=f"Step-{pair_step} Frame Change Distance - {video_name}",
        ylabel="Change Distance = 1 - Cosine Similarity",
        save_path=distance_png,
        threshold=threshold,
        peak_indices=peak_indices,
    )

    # 灰度直方图 Bhattacharyya 距离曲线
    gray_hist_png = os.path.join(video_out_dir, "gray_hist_bhattacharyya_distance_curve.png")
    plot_curve(
        rows=gray_rows,
        y_key="bhattacharyya_distance",
        title=f"Step-{pair_step} Gray Histogram Bhattacharyya Distance - {video_name}",
        ylabel="Bhattacharyya Distance",
        save_path=gray_hist_png,
        threshold=gray_threshold,
        peak_indices=gray_peak_indices,
    )

    # RGB 直方图 Bhattacharyya 距离曲线
    rgb_hist_png = os.path.join(video_out_dir, "rgb_hist_bhattacharyya_distance_curve.png")
    plot_curve(
        rows=rgb_rows,
        y_key="bhattacharyya_distance",
        title=f"Step-{pair_step} RGB Histogram Bhattacharyya Distance - {video_name}",
        ylabel="Bhattacharyya Distance",
        save_path=rgb_hist_png,
        threshold=rgb_threshold,
        peak_indices=rgb_peak_indices,
    )

    print(f"[SAVE] embeddings: {embedding_path}")
    print(f"[SAVE] cosine scores csv: {csv_path}")
    print(f"[SAVE] gray hist scores csv: {gray_csv_path}")
    print(f"[SAVE] rgb hist scores csv: {rgb_csv_path}")
    print(f"[SAVE] cosine curve: {cosine_png}")
    print(f"[SAVE] distance curve: {distance_png}")
    print(f"[SAVE] gray hist distance curve: {gray_hist_png}")
    print(f"[SAVE] rgb hist distance curve: {rgb_hist_png}")
    print(f"[SAVE] candidate boundaries: {boundary_path}")
    print(f"[DONE] {video_name}")


def main():
    ensure_dir(OUTPUT_DIR)

    device = get_device()
    print(f"[DEVICE] {device}")

    processor, model = load_siglip2_model(MODEL_PATH, device)

    for item in VIDEO_LIST:
        try:
            process_one_video(
                video_item=item,
                processor=processor,
                model=model,
                device=device,
            )
        except Exception as e:
            print(f"[ERROR] 处理失败: {item.get('video_path')}")
            print(f"        {repr(e)}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
基于 scene_segmentation 输出结果提取关键帧：
1) 依据候选边界时间，把预置位切换灰色区域去掉；
2) 剩余连续帧作为稳定场景段；
3) 每段使用长度为 window_size 的连续滑动窗口，选启发函数值最高的窗口；
4) 选中窗口均匀降采样到 target_frames 张图。

启发函数：
- 对窗口内 embedding 相邻帧计算动作距离序列 adj = 1 - cosine(e_t, e_{t+1})；
- 计算动作统计量 mean_motion / active_ratio / peak；
- 清晰度作为第四项分数（拉普拉斯方差，窗口内取均值）；
- 四项在段内归一化后按权重加权；权重可设为 None 以跳过对应资源读取与计算。
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_SCENE_ROOT = "/home/xtc/PipeVideo/scene_segmentation_outputs"
DEFAULT_VIDEO_LIST = "/home/xtc/PipeVideo/video_list.json"
DEFAULT_WINDOW_SIZE = 16
DEFAULT_TARGET_FRAMES = 8
DEFAULT_SHADE_PRE_SECONDS = 1.0
DEFAULT_SHADE_POST_SECONDS = 3.0
DEFAULT_OUTPUT_SUBDIR = "key_frames"
DEFAULT_MEAN_MOTION_WEIGHT = 1
DEFAULT_ACTIVE_RATIO_WEIGHT = None
DEFAULT_PEAK_WEIGHT = None
DEFAULT_SHARPNESS_WEIGHT = None
DEFAULT_ACTIVE_RATIO_THRESHOLD = 0.08
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".ts", ".webm")
NONE_STRINGS = {"none", "null", "nil", "na", "n/a"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_video_name(video_path: str) -> str:
    return Path(video_path).stem.replace(" ", "_")


def parse_optional_float(value: str) -> Optional[float]:
    text = str(value).strip().lower()
    if text in NONE_STRINGS:
        return None
    return float(value)


def weight_to_json(weight: Optional[float]) -> Optional[float]:
    return None if weight is None else float(weight)


def build_heuristic_rule(
    mean_motion_weight: Optional[float],
    active_ratio_weight: Optional[float],
    peak_weight: Optional[float],
    sharpness_weight: Optional[float],
) -> str:
    terms = []
    if mean_motion_weight is not None:
        terms.append(f"{float(mean_motion_weight):g} * norm(mean_motion)")
    if active_ratio_weight is not None:
        terms.append(f"{float(active_ratio_weight):g} * norm(active_ratio)")
    if peak_weight is not None:
        terms.append(f"{float(peak_weight):g} * norm(peak)")
    if sharpness_weight is not None:
        terms.append(f"{float(sharpness_weight):g} * norm(sharpness)")
    if not terms:
        return "score = 0"
    return "score = " + " + ".join(terms)


class VideoFrameReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开原视频用于回退读取: {video_path}")
        self.last_index: Optional[int] = None
        self.last_frame: Optional[np.ndarray] = None

    def read_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f"original_frame_index 不能小于 0，当前值: {frame_index}")

        if self.last_index is not None and self.last_frame is not None and frame_index == self.last_index:
            return self.last_frame.copy()

        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        if not ok:
            raise RuntimeError(
                f"无法定位到原视频帧: video={self.video_path}, frame_index={frame_index}"
            )
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"读取原视频帧失败: video={self.video_path}, frame_index={frame_index}"
            )
        self.last_index = int(frame_index)
        self.last_frame = frame
        return frame.copy()

    def close(self) -> None:
        self.cap.release()


class FrameResolver:
    """
    帧来源双模式：
    1) 优先使用 scene_segmentation 保存的 frame_save_path；
    2) 若不存在，则回退到原视频按 original_frame_index 读取。
    """

    def __init__(self, source_video_path: Optional[str]):
        self.source_video_path = source_video_path
        self.reader: Optional[VideoFrameReader] = None
        self.hit_saved = 0
        self.hit_fallback = 0
        self._printed_fallback_banner = False

    def _parse_original_frame_index(self, meta: Dict, sampled_idx: int) -> int:
        if "original_frame_index" not in meta:
            raise RuntimeError(
                f"meta 缺少 original_frame_index，无法回退到原视频读取: sampled_idx={sampled_idx}"
            )
        try:
            return int(meta["original_frame_index"])
        except Exception as e:
            raise RuntimeError(
                f"original_frame_index 非法，无法回退到原视频读取: "
                f"sampled_idx={sampled_idx}, value={meta.get('original_frame_index')}"
            ) from e

    def _get_reader(self) -> VideoFrameReader:
        if self.reader is not None:
            return self.reader
        if not self.source_video_path:
            raise RuntimeError(
                "frame_save_path 不可用，且无法确定原视频路径。"
                "请传 --video-list 以提供 video_path，或在 scene_segmentation 阶段保存 sampled_frames。"
            )
        self.reader = VideoFrameReader(self.source_video_path)
        if not self._printed_fallback_banner:
            print(f"[FALLBACK] use original video for frame loading: {self.source_video_path}")
            self._printed_fallback_banner = True
        return self.reader

    def load_frame_bgr(
        self,
        meta: Dict,
        sampled_idx: int,
    ) -> Tuple[np.ndarray, str, bool]:
        frame_save_path = str(meta.get("frame_save_path", "")).strip()
        if frame_save_path and os.path.isfile(frame_save_path):
            frame = cv2.imread(frame_save_path, cv2.IMREAD_COLOR)
            if frame is not None:
                self.hit_saved += 1
                return frame, frame_save_path, True

        original_frame_index = self._parse_original_frame_index(meta, sampled_idx)
        frame = self._get_reader().read_frame(original_frame_index)
        self.hit_fallback += 1
        source_ref = f"{self.source_video_path}#frame={original_frame_index}"
        return frame, source_ref, False

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None


def build_intervals(
    boundaries: List[Dict],
    pre_seconds: float,
    post_seconds: float,
) -> Tuple[List[Tuple[float, float]], str, Dict[str, int]]:
    """
    区间来源优先级：
    1) boundary 自带 adaptive_left/right_time_sec；
    2) boundary 自带 adaptive_pre/post_seconds；
    3) 回退固定 pre/post 参数。

    同时，pre_seconds / post_seconds 作为最小下限：
    - 当自适应 pre/post 小于下限时，按下限扩展；
    - 当使用固定值时，仍按固定 pre/post。
    """
    intervals: List[Tuple[float, float]] = []
    source_counts = {
        "adaptive_left_right": 0,
        "adaptive_pre_post": 0,
        "fixed_pre_post": 0,
    }
    for item in boundaries:
        t = float(item["time_mid_sec"])
        use_pre = float(pre_seconds)
        use_post = float(post_seconds)

        if ("adaptive_left_time_sec" in item) and ("adaptive_right_time_sec" in item):
            left_raw = float(item["adaptive_left_time_sec"])
            right_raw = float(item["adaptive_right_time_sec"])
            adaptive_pre = max(0.0, t - left_raw)
            adaptive_post = max(0.0, right_raw - t)
            use_pre = max(use_pre, adaptive_pre)
            use_post = max(use_post, adaptive_post)
            source_counts["adaptive_left_right"] += 1
        elif ("adaptive_pre_seconds" in item) and ("adaptive_post_seconds" in item):
            adaptive_pre = max(0.0, float(item["adaptive_pre_seconds"]))
            adaptive_post = max(0.0, float(item["adaptive_post_seconds"]))
            use_pre = max(use_pre, adaptive_pre)
            use_post = max(use_post, adaptive_post)
            source_counts["adaptive_pre_post"] += 1
        else:
            source_counts["fixed_pre_post"] += 1

        left = t - use_pre
        right = t + use_post

        if right < left:
            left, right = right, left
        intervals.append((left, right))

    if not intervals:
        return [], "empty", source_counts

    intervals.sort(key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for left, right in intervals[1:]:
        last_left, last_right = merged[-1]
        if left <= last_right:
            merged[-1] = (last_left, max(last_right, right))
        else:
            merged.append((left, right))

    if source_counts["fixed_pre_post"] == 0:
        if source_counts["adaptive_left_right"] > 0:
            interval_source = "adaptive"
        else:
            interval_source = "adaptive_pre_post"
    elif source_counts["adaptive_left_right"] == 0 and source_counts["adaptive_pre_post"] == 0:
        interval_source = "fixed"
    else:
        interval_source = "mixed"

    return merged, interval_source, source_counts


def mark_gray_frames(
    metas: List[Dict],
    gray_intervals: List[Tuple[float, float]],
) -> List[bool]:
    gray_mask = [False] * len(metas)
    if not gray_intervals:
        return gray_mask

    interval_id = 0
    num_intervals = len(gray_intervals)
    for i, meta in enumerate(metas):
        t = float(meta["timestamp_sec"])
        while interval_id < num_intervals and gray_intervals[interval_id][1] < t:
            interval_id += 1
        if interval_id < num_intervals:
            left, right = gray_intervals[interval_id]
            if left <= t <= right:
                gray_mask[i] = True
    return gray_mask


def split_stable_segments(gray_mask: List[bool]) -> List[List[int]]:
    segments: List[List[int]] = []
    cur: List[int] = []
    for idx, is_gray in enumerate(gray_mask):
        if is_gray:
            if cur:
                segments.append(cur)
                cur = []
        else:
            cur.append(idx)
    if cur:
        segments.append(cur)
    return segments


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return emb / norms


def window_action_features(
    emb_norm: np.ndarray,
    frame_indices: List[int],
    need_mean_motion: bool,
    need_active_ratio: bool,
    need_peak: bool,
    active_ratio_threshold: float = DEFAULT_ACTIVE_RATIO_THRESHOLD,
) -> Dict[str, float]:
    if not (need_mean_motion or need_active_ratio or need_peak):
        return {
            "mean_motion": 0.0,
            "active_ratio": 0.0,
            "peak": 0.0,
        }

    if len(frame_indices) <= 1:
        return {
            "mean_motion": 0.0,
            "active_ratio": 0.0,
            "peak": 0.0,
        }
    lhs = emb_norm[frame_indices[:-1]]
    rhs = emb_norm[frame_indices[1:]]
    adj = 1.0 - np.sum(lhs * rhs, axis=1)
    out = {
        "mean_motion": 0.0,
        "active_ratio": 0.0,
        "peak": 0.0,
    }
    if need_mean_motion:
        out["mean_motion"] = float(np.mean(adj))
    if need_active_ratio:
        out["active_ratio"] = float(np.mean(adj > float(active_ratio_threshold)))
    if need_peak:
        out["peak"] = float(np.max(adj))
    return out


def compute_frame_sharpness_values(
    metas: List[Dict],
    frame_resolver: FrameResolver,
) -> List[float]:
    sharpness_values: List[float] = []
    for i, meta in enumerate(metas):
        frame_bgr, _, _ = frame_resolver.load_frame_bgr(meta=meta, sampled_idx=i)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_values.append(float(lap.var()))
    return sharpness_values


def minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < 1e-12:
        return [1.0 for _ in values]
    out = (arr - vmin) / (vmax - vmin)
    return out.astype(np.float64).tolist()


def window_mean_sharpness(
    frame_sharpness_values: List[float],
    frame_indices: List[int],
) -> float:
    if not frame_indices:
        return 0.0
    vals = [float(frame_sharpness_values[idx]) for idx in frame_indices]
    return float(np.mean(vals))


def select_best_window_in_segment(
    emb_norm: Optional[np.ndarray],
    frame_sharpness_values: Optional[List[float]],
    segment_indices: List[int],
    window_size: int,
    mean_motion_weight: Optional[float] = DEFAULT_MEAN_MOTION_WEIGHT,
    active_ratio_weight: Optional[float] = DEFAULT_ACTIVE_RATIO_WEIGHT,
    peak_weight: Optional[float] = DEFAULT_PEAK_WEIGHT,
    sharpness_weight: Optional[float] = DEFAULT_SHARPNESS_WEIGHT,
    active_ratio_threshold: float = DEFAULT_ACTIVE_RATIO_THRESHOLD,
) -> Tuple[List[int], Dict]:
    need_mean_motion = mean_motion_weight is not None
    need_active_ratio = active_ratio_weight is not None
    need_peak = peak_weight is not None
    need_action = need_mean_motion or need_active_ratio or need_peak
    need_sharpness = sharpness_weight is not None

    if need_action and emb_norm is None:
        raise RuntimeError("动作特征权重启用时，emb_norm 不能为空。")
    if need_sharpness and frame_sharpness_values is None:
        raise RuntimeError("清晰度权重启用时，frame_sharpness_values 不能为空。")

    seg_len = len(segment_indices)
    if seg_len == 0:
        return [], {
            "best_offset": -1,
            "heuristic_score": 0.0,
            "mean_motion": None if not need_mean_motion else 0.0,
            "active_ratio": None if not need_active_ratio else 0.0,
            "peak": None if not need_peak else 0.0,
            "sharpness": None if not need_sharpness else 0.0,
            "num_candidate_windows": 0,
        }

    candidate_windows: List[List[int]] = []
    candidate_offsets: List[int] = []
    if seg_len < window_size:
        candidate_windows = [list(segment_indices)]
        candidate_offsets = [0]
    else:
        max_start = seg_len - window_size
        for offset in range(max_start + 1):
            candidate_windows.append(segment_indices[offset : offset + window_size])
            candidate_offsets.append(offset)

    num_windows = len(candidate_windows)
    zeros = [0.0 for _ in range(num_windows)]

    if need_action:
        action_raw = [
            window_action_features(
                emb_norm,
                w,
                need_mean_motion=need_mean_motion,
                need_active_ratio=need_active_ratio,
                need_peak=need_peak,
                active_ratio_threshold=active_ratio_threshold,
            )
            for w in candidate_windows
        ]
    else:
        action_raw = []

    mean_motion_raw = [f["mean_motion"] for f in action_raw] if need_mean_motion else zeros
    active_ratio_raw = [f["active_ratio"] for f in action_raw] if need_active_ratio else zeros
    peak_raw = [f["peak"] for f in action_raw] if need_peak else zeros
    sharp_raw = (
        [window_mean_sharpness(frame_sharpness_values, w) for w in candidate_windows]
        if need_sharpness
        else zeros
    )

    mean_motion_norm = minmax_normalize(mean_motion_raw) if need_mean_motion else zeros
    active_ratio_norm = minmax_normalize(active_ratio_raw) if need_active_ratio else zeros
    peak_norm = minmax_normalize(peak_raw) if need_peak else zeros
    sharp_norm = minmax_normalize(sharp_raw) if need_sharpness else zeros

    combined: List[float] = []
    for i in range(num_windows):
        score = 0.0
        if need_mean_motion:
            score += float(mean_motion_weight) * float(mean_motion_norm[i])
        if need_active_ratio:
            score += float(active_ratio_weight) * float(active_ratio_norm[i])
        if need_peak:
            score += float(peak_weight) * float(peak_norm[i])
        if need_sharpness:
            score += float(sharpness_weight) * float(sharp_norm[i])
        combined.append(float(score))

    best_id = int(np.argmax(np.array(combined, dtype=np.float64)))
    best_window = candidate_windows[best_id]

    best_info = {
        "best_offset": int(candidate_offsets[best_id]),
        "heuristic_score": float(combined[best_id]),
        "mean_motion": float(mean_motion_norm[best_id]) if need_mean_motion else None,
        "active_ratio": float(active_ratio_norm[best_id]) if need_active_ratio else None,
        "peak": float(peak_norm[best_id]) if need_peak else None,
        "sharpness": float(sharp_norm[best_id]) if need_sharpness else None,
        "num_candidate_windows": len(candidate_windows),
    }
    return best_window, best_info


def uniform_sample_indices(indices: List[int], target_count: int) -> List[int]:
    if not indices:
        return []
    if target_count <= 1:
        return [indices[0]]
    pos = np.linspace(0, len(indices) - 1, num=target_count)
    rel = np.rint(pos).astype(int).tolist()
    return [indices[i] for i in rel]


def choose_video_dirs(scene_root: str, video_list_path: str = "") -> List[str]:
    root = Path(scene_root)
    if not root.exists():
        raise FileNotFoundError(f"scene_root 不存在: {scene_root}")

    if video_list_path:
        data = load_json(video_list_path)
        names = [safe_video_name(item["video_path"]) for item in data]
        dirs = [str(root / name) for name in names]
    else:
        dirs = [str(p) for p in sorted(root.iterdir()) if p.is_dir()]
    return dirs


def build_video_name_to_path(video_list_path: str) -> Dict[str, str]:
    if not video_list_path:
        return {}
    data = load_json(video_list_path)
    if not isinstance(data, list):
        raise ValueError(f"video_list 必须是 list，当前类型: {type(data)}")

    mapping: Dict[str, str] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "video_path" not in item:
            raise ValueError(f"video_list[{idx}] 格式错误，必须包含 video_path 字段")
        video_path = str(item["video_path"])
        mapping.setdefault(safe_video_name(video_path), video_path)
    return mapping


def infer_video_path_from_dirname(video_name: str) -> Optional[str]:
    default_video_root = Path("/home/xtc/PipeVideo/video_data")
    for ext in VIDEO_EXTENSIONS:
        candidate = default_video_root / f"{video_name}{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


def process_one_video_dir(
    video_dir: str,
    source_video_path: Optional[str],
    mean_motion_weight: Optional[float],
    active_ratio_weight: Optional[float],
    peak_weight: Optional[float],
    sharpness_weight: Optional[float],
    pre_seconds: float,
    post_seconds: float,
    window_size: int,
    target_frames: int,
    output_subdir: str,
) -> Dict:
    candidate_path = os.path.join(video_dir, "candidate_boundaries.json")
    meta_path = os.path.join(video_dir, "sampled_frames_meta.json")
    emb_path = os.path.join(video_dir, "frame_embeddings.npy")

    if not os.path.isfile(candidate_path):
        raise FileNotFoundError(f"缺少文件: {candidate_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"缺少文件: {meta_path}")

    candidate = load_json(candidate_path)
    metas = load_json(meta_path)

    need_mean_motion = mean_motion_weight is not None
    need_active_ratio = active_ratio_weight is not None
    need_peak = peak_weight is not None
    need_action = need_mean_motion or need_active_ratio or need_peak
    need_sharpness = sharpness_weight is not None

    if not (need_action or need_sharpness):
        raise ValueError("四项权重不能同时为 None，至少需要启用一项。")

    emb: Optional[np.ndarray] = None
    if need_action:
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(f"缺少文件: {emb_path}")
        emb = np.load(emb_path)

    if len(metas) == 0:
        raise RuntimeError(f"没有 sampled frames: {video_dir}")
    if emb is not None and emb.shape[0] != len(metas):
        raise RuntimeError(
            f"embedding 数量与 sampled frames 数量不一致: emb={emb.shape[0]}, metas={len(metas)}"
        )

    boundaries = candidate.get("candidate_boundaries", [])
    gray_intervals, interval_source, interval_source_counts = build_intervals(
        boundaries,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )
    gray_mask = mark_gray_frames(metas, gray_intervals)
    stable_segments = split_stable_segments(gray_mask)

    # 如果全部被灰区覆盖，兜底为整段，避免无输出。
    if not stable_segments:
        stable_segments = [list(range(len(metas)))]

    emb_norm = normalize_embeddings(emb) if emb is not None else None
    frame_resolver = FrameResolver(source_video_path=source_video_path)
    try:
        frame_sharpness_values = (
            compute_frame_sharpness_values(
                metas=metas,
                frame_resolver=frame_resolver,
            )
            if need_sharpness
            else None
        )

        out_dir = os.path.join(video_dir, output_subdir)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
            print(f"[CLEAN] removed existing output dir: {out_dir}")
        ensure_dir(out_dir)

        segment_results = []
        total_saved = 0
        for seg_id, seg_indices in enumerate(stable_segments, start=1):
            best_window, best_info = select_best_window_in_segment(
                emb_norm=emb_norm,
                frame_sharpness_values=frame_sharpness_values,
                segment_indices=seg_indices,
                window_size=window_size,
                mean_motion_weight=mean_motion_weight,
                active_ratio_weight=active_ratio_weight,
                peak_weight=peak_weight,
                sharpness_weight=sharpness_weight,
                active_ratio_threshold=DEFAULT_ACTIVE_RATIO_THRESHOLD,
            )
            sampled = uniform_sample_indices(best_window, target_frames)

            seg_dir = os.path.join(out_dir, f"segment_{seg_id:03d}")
            ensure_dir(seg_dir)

            sampled_items = []
            for rank, frame_idx in enumerate(sampled, start=1):
                meta = metas[frame_idx]
                frame_bgr, source_ref, from_saved_image = frame_resolver.load_frame_bgr(
                    meta=meta,
                    sampled_idx=frame_idx,
                )
                if from_saved_image:
                    src_name = os.path.basename(source_ref)
                    dst_name = f"rank_{rank:02d}_idx_{frame_idx:06d}_{src_name}"
                    dst_path = os.path.join(seg_dir, dst_name)
                    shutil.copy2(source_ref, dst_path)
                else:
                    original_frame_index = int(meta["original_frame_index"])
                    dst_name = (
                        f"rank_{rank:02d}_idx_{frame_idx:06d}_orig_{original_frame_index:08d}.jpg"
                    )
                    dst_path = os.path.join(seg_dir, dst_name)
                    ok = cv2.imwrite(dst_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise RuntimeError(f"保存关键帧失败: {dst_path}")
                total_saved += 1

                sampled_items.append(
                    {
                        "rank": rank,
                        "frame_index": int(frame_idx),
                        "timestamp_sec": float(meta["timestamp_sec"]),
                        "source_path": source_ref,
                        "saved_path": dst_path,
                    }
                )

            seg_meta = {
                "segment_id": seg_id,
                "segment_start_index": int(seg_indices[0]),
                "segment_end_index": int(seg_indices[-1]),
                "segment_num_frames": len(seg_indices),
                "best_window_start_index": int(best_window[0]) if best_window else None,
                "best_window_end_index": int(best_window[-1]) if best_window else None,
                "best_window_num_frames": len(best_window),
                "best_window_offset_in_segment": int(best_info["best_offset"]),
                "heuristic_score": float(best_info["heuristic_score"]),
                "heuristic_score_mean_motion": weight_to_json(best_info["mean_motion"]),
                "heuristic_score_active_ratio": weight_to_json(best_info["active_ratio"]),
                "heuristic_score_peak": weight_to_json(best_info["peak"]),
                "heuristic_score_sharpness": weight_to_json(best_info["sharpness"]),
                "num_candidate_windows": int(best_info["num_candidate_windows"]),
                "uniform_sample_target_count": int(target_frames),
                "sampled_frames": sampled_items,
            }
            segment_results.append(seg_meta)

        result = {
            "video_dir": video_dir,
            "source_video_path": source_video_path or "",
            "window_size": int(window_size),
            "target_frames": int(target_frames),
            "heuristic_weights": {
                "mean_motion": weight_to_json(mean_motion_weight),
                "active_ratio": weight_to_json(active_ratio_weight),
                "peak": weight_to_json(peak_weight),
                "sharpness": weight_to_json(sharpness_weight),
            },
            "active_ratio_threshold": (
                float(DEFAULT_ACTIVE_RATIO_THRESHOLD) if need_active_ratio else None
            ),
            "heuristic_rule": build_heuristic_rule(
                mean_motion_weight=mean_motion_weight,
                active_ratio_weight=active_ratio_weight,
                peak_weight=peak_weight,
                sharpness_weight=sharpness_weight,
            ),
            "shade_pre_seconds": float(pre_seconds),
            "shade_post_seconds": float(post_seconds),
            "gray_interval_source": interval_source,
            "gray_interval_source_counts": interval_source_counts,
            "gray_intervals": [
                {"start_sec": float(s), "end_sec": float(e)} for s, e in gray_intervals
            ],
            "num_total_sampled_frames": len(metas),
            "num_gray_frames": int(sum(gray_mask)),
            "num_stable_segments": len(stable_segments),
            "segments": segment_results,
            "num_saved_key_frames": int(total_saved),
            "num_source_hits_saved_image": int(frame_resolver.hit_saved),
            "num_source_hits_video_fallback": int(frame_resolver.hit_fallback),
            "resource_usage": {
                "loaded_embeddings": bool(need_action),
                "computed_sharpness_from_frames": bool(need_sharpness),
            },
        }

        save_json(result, os.path.join(out_dir, "key_frame_selection.json"))
        return result
    finally:
        frame_resolver.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 scene_segmentation 输出上提取每个稳定场景段的关键帧。"
    )
    parser.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help=f"scene_segmentation 输出根目录，默认: {DEFAULT_SCENE_ROOT}",
    )
    parser.add_argument(
        "--video-list",
        default=DEFAULT_VIDEO_LIST,
        help=(
            "视频列表 JSON（用于限定处理哪些视频）；"
            "传空字符串可处理 scene-root 下全部子目录。"
        ),
    )
    parser.add_argument(
        "--shade-pre-seconds",
        type=float,
        default=DEFAULT_SHADE_PRE_SECONDS,
        help=(
            f"灰区左侧最小时长（秒），默认: {DEFAULT_SHADE_PRE_SECONDS}。"
            "当自适应 pre 更小时，会用该值作为下限。"
        ),
    )
    parser.add_argument(
        "--shade-post-seconds",
        type=float,
        default=DEFAULT_SHADE_POST_SECONDS,
        help=(
            f"灰区右侧最小时长（秒），默认: {DEFAULT_SHADE_POST_SECONDS}。"
            "当自适应 post 更小时，会用该值作为下限。"
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"滑动窗口长度，默认: {DEFAULT_WINDOW_SIZE}",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=f"每个段输出帧数，默认: {DEFAULT_TARGET_FRAMES}",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help=f"输出子目录名（位于每个视频目录下），默认: {DEFAULT_OUTPUT_SUBDIR}",
    )
    parser.add_argument(
        "--mean-motion-weight",
        type=parse_optional_float,
        default=DEFAULT_MEAN_MOTION_WEIGHT,
        help=(
            f"mean_motion 权重，默认: {DEFAULT_MEAN_MOTION_WEIGHT}；"
            "可设为 none 以跳过该项计算。"
        ),
    )
    parser.add_argument(
        "--active-ratio-weight",
        type=parse_optional_float,
        default=DEFAULT_ACTIVE_RATIO_WEIGHT,
        help=(
            f"active_ratio 权重，默认: {DEFAULT_ACTIVE_RATIO_WEIGHT}；"
            "可设为 none 以跳过该项计算。"
        ),
    )
    parser.add_argument(
        "--peak-weight",
        type=parse_optional_float,
        default=DEFAULT_PEAK_WEIGHT,
        help=f"peak 权重，默认: {DEFAULT_PEAK_WEIGHT}；可设为 none 以跳过该项计算。",
    )
    parser.add_argument(
        "--sharpness-weight",
        type=parse_optional_float,
        default=DEFAULT_SHARPNESS_WEIGHT,
        help=(
            f"sharpness 权重，默认: {DEFAULT_SHARPNESS_WEIGHT}；"
            "可设为 none 以跳过清晰度资源读取与计算。"
        ),
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help=(
            "summary 输出路径；默认空字符串，表示写入 "
            "scene_root/key_frame_selection_summary.json。"
        ),
    )
    return parser


def main():
    args = build_argparser().parse_args()

    if args.window_size <= 0:
        raise ValueError("--window-size 必须大于 0")
    if args.target_frames <= 0:
        raise ValueError("--target-frames 必须大于 0")
    if args.shade_pre_seconds < 0 or args.shade_post_seconds < 0:
        raise ValueError("--shade-pre-seconds / --shade-post-seconds 不能小于 0")
    if (
        args.mean_motion_weight is None
        and args.active_ratio_weight is None
        and args.peak_weight is None
        and args.sharpness_weight is None
    ):
        raise ValueError("四项权重不能同时为 None，至少保留一项。")

    video_list_path = args.video_list.strip()
    video_dirs = choose_video_dirs(
        scene_root=args.scene_root,
        video_list_path=video_list_path,
    )
    if not video_dirs:
        raise RuntimeError("没有可处理的视频目录")
    video_name_to_path = build_video_name_to_path(video_list_path)

    summary = []
    for video_dir in video_dirs:
        try:
            video_name = Path(video_dir).name
            source_video_path = video_name_to_path.get(video_name) or infer_video_path_from_dirname(
                video_name
            )
            result = process_one_video_dir(
                video_dir=video_dir,
                source_video_path=source_video_path,
                mean_motion_weight=args.mean_motion_weight,
                active_ratio_weight=args.active_ratio_weight,
                peak_weight=args.peak_weight,
                sharpness_weight=args.sharpness_weight,
                pre_seconds=args.shade_pre_seconds,
                post_seconds=args.shade_post_seconds,
                window_size=args.window_size,
                target_frames=args.target_frames,
                output_subdir=args.output_subdir,
            )
            summary.append(
                {
                    "video_dir": video_dir,
                    "status": "ok",
                    "num_stable_segments": result["num_stable_segments"],
                    "num_saved_key_frames": result["num_saved_key_frames"],
                    "num_source_hits_saved_image": result["num_source_hits_saved_image"],
                    "num_source_hits_video_fallback": result["num_source_hits_video_fallback"],
                    "resource_usage": result["resource_usage"],
                }
            )
            print(
                f"[DONE] {video_dir} | segments={result['num_stable_segments']} "
                f"| saved_key_frames={result['num_saved_key_frames']} "
                f"| source(saved={result['num_source_hits_saved_image']}, "
                f"fallback={result['num_source_hits_video_fallback']})"
            )
        except Exception as e:
            summary.append(
                {
                    "video_dir": video_dir,
                    "status": "error",
                    "error": repr(e),
                }
            )
            print(f"[ERROR] {video_dir} | {repr(e)}")

    summary_path = args.summary_path.strip() or os.path.join(
        args.scene_root, "key_frame_selection_summary.json"
    )
    save_json(summary, summary_path)
    print(f"[SAVE] summary -> {summary_path}")


if __name__ == "__main__":
    main()

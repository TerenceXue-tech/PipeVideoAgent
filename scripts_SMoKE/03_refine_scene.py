# -*- coding: utf-8 -*-
"""
场景精炼脚本：
- 输入：scene_segmentation_outputs/<video>/key_frames/key_frame_selection.json
- 目标：避免同一场景被切成多个相邻段。
- 方法：
  1) 相邻场景的关键帧两两计算 embedding cosine 相似度，取最大值；
  2) 若该最大相似度 >= 阈值，判为同一场景；
  3) 同一场景候选中只保留“内部差异度”更高者；
  4) 删除另一个场景并按顺序重命名为 segment_001, segment_002 ...

说明：
- 关键帧 embedding 直接使用 scene_segmentation 阶段保存的 frame_embeddings.npy，
  通过 sampled_frames[*].frame_index 索引；不重复跑视觉模型。
- 内部差异度定义为：场景关键帧相邻距离均值 mean(1 - cosine(e_t, e_{t+1}))。
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_SCENE_ROOT = "/home/xtc/PipeVideo/scene_segmentation_outputs"
DEFAULT_VIDEO_LIST = "/home/xtc/PipeVideo/video_list.json"
DEFAULT_INPUT_SUBDIR = "key_frames"
DEFAULT_OUTPUT_SUBDIR = "key_frames_refined"
DEFAULT_KEY_FRAME_JSON_NAME = "key_frame_selection.json"
DEFAULT_EMBEDDING_NAME = "frame_embeddings.npy"
DEFAULT_REPORT_NAME = "refine_scene_report.json"
DEFAULT_SUMMARY_NAME = "refine_scene_summary.json"
DEFAULT_SIMILARITY_THRESHOLD = 0.98


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_video_name(video_path: str) -> str:
    return Path(video_path).stem.replace(" ", "_")


def choose_video_dirs(scene_root: str, video_list_path: str = "") -> List[str]:
    root = Path(scene_root)
    if not root.exists():
        raise FileNotFoundError(f"scene_root 不存在: {scene_root}")

    if video_list_path:
        data = load_json(video_list_path)
        names = [safe_video_name(item["video_path"]) for item in data]
        return [str(root / name) for name in names]

    return [str(p) for p in sorted(root.iterdir()) if p.is_dir()]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return emb / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class SegmentStat:
    segment: Dict
    original_segment_id: int
    key_frame_indices: List[int]
    key_frame_embeddings: np.ndarray
    internal_diversity: float


def extract_key_frame_indices(sampled_frames: List[Dict]) -> List[int]:
    if not sampled_frames:
        raise RuntimeError("sampled_frames 为空，无法提取关键帧索引")
    return [int(item["frame_index"]) for item in sampled_frames]


def calc_internal_diversity(indices: List[int], emb_norm: np.ndarray) -> float:
    """
    窗口内部差异度：关键帧相邻距离均值。
    若不足两帧，返回 0。
    """
    if len(indices) <= 1:
        return 0.0

    for idx in indices:
        if idx < 0 or idx >= emb_norm.shape[0]:
            raise RuntimeError(f"frame_index 越界: {idx}, embedding_count={emb_norm.shape[0]}")

    values = []
    for i in range(len(indices) - 1):
        s = cosine_sim(emb_norm[indices[i]], emb_norm[indices[i + 1]])
        values.append(1.0 - s)
    return float(np.mean(values)) if values else 0.0


def build_segment_stats(segments: List[Dict], emb_norm: np.ndarray) -> List[SegmentStat]:
    stats: List[SegmentStat] = []
    for seg in segments:
        sampled = seg.get("sampled_frames", [])
        if not sampled:
            raise RuntimeError(f"segment 缺少 sampled_frames: segment_id={seg.get('segment_id')}")

        indices = extract_key_frame_indices(sampled)
        for idx in indices:
            if idx < 0 or idx >= emb_norm.shape[0]:
                raise RuntimeError(f"frame_index 越界: {idx}, embedding_count={emb_norm.shape[0]}")

        key_emb = emb_norm[np.asarray(indices, dtype=np.int64)]
        diversity = calc_internal_diversity(indices, emb_norm)
        stats.append(
            SegmentStat(
                segment=seg,
                original_segment_id=int(seg.get("segment_id", len(stats) + 1)),
                key_frame_indices=indices,
                key_frame_embeddings=key_emb,
                internal_diversity=diversity,
            )
        )
    return stats


def calc_pair_max_similarity(
    left: SegmentStat,
    right: SegmentStat,
) -> Tuple[float, int, int]:
    """
    计算两个场景关键帧两两余弦相似度最大值，返回：
    - 最大相似度
    - left 场景命中关键帧在 key_frame_indices 中的位置
    - right 场景命中关键帧在 key_frame_indices 中的位置
    """
    sim_mat = left.key_frame_embeddings @ right.key_frame_embeddings.T
    flat_idx = int(np.argmax(sim_mat))
    li, ri = np.unravel_index(flat_idx, sim_mat.shape)
    return float(sim_mat[li, ri]), int(li), int(ri)


def refine_segments(
    stats: List[SegmentStat],
    similarity_threshold: float,
) -> Tuple[List[SegmentStat], List[Dict]]:
    """
    对相邻场景做相似性去重，返回保留的场景和决策日志。
    """
    kept = list(stats)
    decisions: List[Dict] = []

    i = 0
    while i < len(kept) - 1:
        left = kept[i]
        right = kept[i + 1]

        sim, li, ri = calc_pair_max_similarity(left, right)
        if sim < similarity_threshold:
            i += 1
            continue

        # 相似场景：保留内部差异度更高者
        if left.internal_diversity >= right.internal_diversity:
            keep_idx = i
            drop_idx = i + 1
        else:
            keep_idx = i + 1
            drop_idx = i

        keep_seg = kept[keep_idx]
        drop_seg = kept[drop_idx]

        decisions.append(
            {
                "left_original_segment_id": left.original_segment_id,
                "right_original_segment_id": right.original_segment_id,
                "max_pair_similarity": float(sim),
                "left_matched_frame_index": int(left.key_frame_indices[li]),
                "right_matched_frame_index": int(right.key_frame_indices[ri]),
                "left_internal_diversity": float(left.internal_diversity),
                "right_internal_diversity": float(right.internal_diversity),
                "kept_original_segment_id": keep_seg.original_segment_id,
                "dropped_original_segment_id": drop_seg.original_segment_id,
                "reason": "similar_scene_keep_higher_internal_diversity",
            }
        )

        del kept[drop_idx]
        i = max(0, i - 1)

    return kept, decisions


def remap_segment_dir_name(new_segment_id: int) -> str:
    return f"segment_{new_segment_id:03d}"


def copy_and_rewrite_segment(
    segment: Dict,
    new_segment_id: int,
    output_dir: str,
) -> Dict:
    """
    复制 segment 图片目录并重写 sampled_frames[*].saved_path。
    """
    new_seg = dict(segment)
    new_seg["segment_id"] = int(new_segment_id)

    new_seg_dir = os.path.join(output_dir, remap_segment_dir_name(new_segment_id))
    ensure_dir(new_seg_dir)

    sampled_out = []
    for item in segment.get("sampled_frames", []):
        new_item = dict(item)

        saved_path = str(item.get("saved_path", "")).strip()
        if not saved_path:
            raise RuntimeError(
                f"sampled_frame 缺少 saved_path: segment_id={segment.get('segment_id')}"
            )
        if not os.path.isfile(saved_path):
            raise FileNotFoundError(f"关键帧文件不存在: {saved_path}")

        dst_name = os.path.basename(saved_path)
        dst_path = os.path.join(new_seg_dir, dst_name)
        shutil.copy2(saved_path, dst_path)

        new_item["saved_path"] = dst_path
        sampled_out.append(new_item)

    new_seg["sampled_frames"] = sampled_out
    return new_seg


def apply_refine_to_video_dir(
    video_dir: str,
    input_subdir: str,
    output_subdir: str,
    key_frame_json_name: str,
    embedding_name: str,
    report_name: str,
    similarity_threshold: float,
    dry_run: bool,
) -> Dict:
    in_dir = os.path.join(video_dir, input_subdir)
    in_json = os.path.join(in_dir, key_frame_json_name)
    emb_path = os.path.join(video_dir, embedding_name)

    if not os.path.isfile(in_json):
        raise FileNotFoundError(f"缺少文件: {in_json}")
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"缺少文件: {emb_path}")

    payload = load_json(in_json)
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or len(segments) == 0:
        raise RuntimeError(f"segments 为空或格式错误: {in_json}")

    emb = np.load(emb_path)
    emb_norm = normalize_embeddings(emb)
    stats = build_segment_stats(segments=segments, emb_norm=emb_norm)

    kept_stats, decisions = refine_segments(
        stats=stats,
        similarity_threshold=float(similarity_threshold),
    )

    original_count = len(stats)
    kept_count = len(kept_stats)
    dropped_count = original_count - kept_count

    report = {
        "video_dir": video_dir,
        "input_subdir": input_subdir,
        "output_subdir": output_subdir,
        "similarity_threshold": float(similarity_threshold),
        "original_num_scenes": int(original_count),
        "refined_num_scenes": int(kept_count),
        "dropped_num_scenes": int(dropped_count),
        "decisions": decisions,
    }

    if dry_run:
        return {
            "video_dir": video_dir,
            "status": "ok_dry_run",
            **report,
        }

    out_dir = os.path.join(video_dir, output_subdir)

    # 同目录输出时，先写到临时目录再原子替换
    inplace = os.path.abspath(in_dir) == os.path.abspath(out_dir)
    work_out_dir = out_dir + ".refine_tmp" if inplace else out_dir

    if os.path.isdir(work_out_dir):
        shutil.rmtree(work_out_dir)
    ensure_dir(work_out_dir)

    new_segments = []
    for new_id, seg_stat in enumerate(kept_stats, start=1):
        new_seg = copy_and_rewrite_segment(
            segment=seg_stat.segment,
            new_segment_id=new_id,
            output_dir=work_out_dir,
        )
        new_segments.append(new_seg)

    new_payload = dict(payload)
    new_payload["segments"] = new_segments
    new_payload["num_stable_segments"] = int(len(new_segments))
    new_payload["num_saved_key_frames"] = int(
        sum(len(seg.get("sampled_frames", [])) for seg in new_segments)
    )
    new_payload["refine_scene"] = {
        "enabled": True,
        "similarity_threshold": float(similarity_threshold),
        "original_num_scenes": int(original_count),
        "refined_num_scenes": int(kept_count),
        "dropped_num_scenes": int(dropped_count),
        "report_name": report_name,
    }

    save_json(new_payload, os.path.join(work_out_dir, key_frame_json_name))
    save_json(report, os.path.join(work_out_dir, report_name))

    if inplace:
        shutil.rmtree(out_dir)
        os.replace(work_out_dir, out_dir)

    return {
        "video_dir": video_dir,
        "status": "ok",
        **report,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据相邻场景关键帧两两最大相似度精炼场景，删除重复切分并重命名场景。"
    )
    parser.add_argument(
        "--scene-root",
        default=DEFAULT_SCENE_ROOT,
        help=f"scene 输出根目录，默认: {DEFAULT_SCENE_ROOT}",
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
        "--input-subdir",
        default=DEFAULT_INPUT_SUBDIR,
        help=f"输入 key_frames 子目录名，默认: {DEFAULT_INPUT_SUBDIR}",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help=(
            f"输出子目录名（位于每个视频目录下），默认: {DEFAULT_OUTPUT_SUBDIR}。"
            "与 input-subdir 相同时原位更新（先临时目录再替换）。"
        ),
    )
    parser.add_argument(
        "--key-frame-json-name",
        default=DEFAULT_KEY_FRAME_JSON_NAME,
        help=f"关键帧元数据文件名，默认: {DEFAULT_KEY_FRAME_JSON_NAME}",
    )
    parser.add_argument(
        "--embedding-name",
        default=DEFAULT_EMBEDDING_NAME,
        help=f"视频目录下 embedding 文件名，默认: {DEFAULT_EMBEDDING_NAME}",
    )
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help=f"每视频 refine 报告文件名，默认: {DEFAULT_REPORT_NAME}",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"相邻场景关键帧两两最大 cosine 相似度阈值，默认: {DEFAULT_SIMILARITY_THRESHOLD}",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help=(
            "summary 输出路径；默认空字符串，表示写入 "
            "scene_root/refine_scene_summary.json。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅计算和打印结果，不落盘修改文件。",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    if not (-1.0 <= args.similarity_threshold <= 1.0):
        raise ValueError("--similarity-threshold 必须在 [-1, 1] 区间")

    video_list_path = args.video_list.strip()
    video_dirs = choose_video_dirs(scene_root=args.scene_root, video_list_path=video_list_path)
    if not video_dirs:
        raise RuntimeError("没有可处理的视频目录")

    summary = []
    for video_dir in video_dirs:
        try:
            result = apply_refine_to_video_dir(
                video_dir=video_dir,
                input_subdir=args.input_subdir,
                output_subdir=args.output_subdir,
                key_frame_json_name=args.key_frame_json_name,
                embedding_name=args.embedding_name,
                report_name=args.report_name,
                similarity_threshold=args.similarity_threshold,
                dry_run=bool(args.dry_run),
            )
            summary.append(result)
            print(
                f"[DONE] {video_dir} | scenes {result['original_num_scenes']} -> {result['refined_num_scenes']} "
                f"| dropped={result['dropped_num_scenes']} | status={result['status']}"
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

    summary_path = args.summary_path.strip() or os.path.join(args.scene_root, DEFAULT_SUMMARY_NAME)
    save_json(summary, summary_path)
    print(f"[SAVE] summary -> {summary_path}")


if __name__ == "__main__":
    main()

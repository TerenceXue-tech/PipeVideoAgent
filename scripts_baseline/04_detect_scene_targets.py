# -*- coding: utf-8 -*-
"""
对 scene_segmentation_outputs 中每个场景的关键帧做目标检测，
输出“场景是否有目标”的 JSON 结果。

输入约定：
- 每个视频目录下存在 key_frames/key_frame_selection.json
- 该文件内 segments[*].sampled_frames[*] 包含关键帧路径（saved_path/source_path）

输出：
1) 每个视频目录一个 scene_target_presence.json
2) scene_root 下一个 scene_target_presence_summary.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch


# =========================
# 默认配置区（直接改这里即可）
# =========================
DEFAULT_SCENE_ROOT = "/home/xtc/PipeVideo/scene_segmentation_outputs"
DEFAULT_KEY_FRAME_SUBDIR = "key_frames_refined"
DEFAULT_KEY_FRAME_JSON_NAME = "key_frame_selection.json"
DEFAULT_OUTPUT_JSON_NAME = "scene_target_presence.json"
DEFAULT_SUMMARY_JSON_NAME = "scene_target_presence_summary.json"
DEFAULT_OBJECT_DETECTION_CODE_DIR = "/home/xtc/PipeVideo/object_detection/code"
DEFAULT_WEIGHTS = "/home/xtc/PipeVideo/object_detection/code/model/best_day.pt"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_IMG_SIZE = 960
DEFAULT_CONF_THRES = 0.4
DEFAULT_IOU_THRES = 0.4
DEFAULT_MAX_DET = 300
DEFAULT_TARGET_CLASSES = ""
DEFAULT_MAX_VIDEOS = 0
DEFAULT_SCENE_IMAGE_PREFIX = "scene"


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


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict],
) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("box", [0, 0, 0, 0])]
        cls_name = str(det.get("class_name", "obj"))
        score = float(det.get("score", 0.0))
        label = f"{cls_name} {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y_top + th + baseline + 4), (0, 220, 0), -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y_top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return out


def parse_target_classes(text: str) -> Optional[Set[str]]:
    text = str(text).strip()
    if not text:
        return None
    items = [x.strip() for x in text.split(",")]
    items = [x for x in items if x]
    if not items:
        return None
    return set(items)


def resolve_frame_path(frame_item: Dict) -> Optional[str]:
    for key in ("saved_path", "source_path", "frame_path"):
        path = str(frame_item.get(key, "")).strip()
        if path and os.path.isfile(path):
            return path
    return None


def find_key_frame_jsons(
    scene_root: str,
    key_frame_subdir: str,
    key_frame_json_name: str,
) -> List[str]:
    root = Path(scene_root)
    if not root.exists():
        raise FileNotFoundError(f"scene_root 不存在: {scene_root}")

    # 支持 scene_root 直接指向单个视频目录
    direct_candidate = root / key_frame_subdir / key_frame_json_name
    if direct_candidate.is_file():
        return [str(direct_candidate)]

    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        candidate = p / key_frame_subdir / key_frame_json_name
        if candidate.is_file():
            out.append(str(candidate))
    return out


def load_yolo_dependencies(object_detection_code_dir: str):
    code_dir = Path(object_detection_code_dir).resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(f"object_detection 代码目录不存在: {code_dir}")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from models.experimental import attempt_load  # noqa
    from utils.augmentations import letterbox  # noqa
    from utils.general import non_max_suppression, scale_boxes  # noqa
    from utils.torch_utils import select_device  # noqa

    return attempt_load, letterbox, non_max_suppression, scale_boxes, select_device


class YoloSceneDetector:
    def __init__(
        self,
        object_detection_code_dir: str,
        weights: str,
        device: str,
        imgsz: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ):
        (
            self.attempt_load,
            self.letterbox,
            self.non_max_suppression,
            self.scale_boxes,
            self.select_device,
        ) = load_yolo_dependencies(object_detection_code_dir)

        self.device = self.select_device(device)
        self.half = self.device.type != "cpu"
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.weights = str(weights)

        self.model = self.attempt_load(self.weights)
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32

    @torch.inference_mode()
    def detect(self, image_bgr: np.ndarray, target_classes: Optional[Set[str]] = None) -> List[Dict]:
        img = self.letterbox(image_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = pred.float()
        pred = self.non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )

        det = pred[0]
        outputs: List[Dict] = []
        if len(det):
            det[:, :4] = self.scale_boxes(img.shape[2:], det[:, :4], image_bgr.shape).round()
            for row in det:
                x1, y1, x2, y2, score, cls_id = row.tolist()
                cls_id_int = int(cls_id)
                cls_name = str(self.names[cls_id_int])
                if target_classes is not None and cls_name not in target_classes:
                    continue

                outputs.append(
                    {
                        "class_id": cls_id_int,
                        "class_name": cls_name,
                        "score": float(score),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )
        return outputs

    def detect_image_path(self, image_path: str, target_classes: Optional[Set[str]] = None) -> List[Dict]:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"读取图片失败: {image_path}")
        return self.detect(image_bgr=image_bgr, target_classes=target_classes)


def process_one_video(
    key_frame_json_path: str,
    detector: YoloSceneDetector,
    target_classes: Optional[Set[str]],
    output_json_name: str,
    scene_image_prefix: str,
) -> Dict:
    key_frame_json = load_json(key_frame_json_path)
    video_dir = str(Path(key_frame_json_path).parent.parent)
    segments = key_frame_json.get("segments", [])

    scene_results = []
    num_scenes_with_target = 0

    for segment in segments:
        seg_id = int(segment.get("segment_id", len(scene_results) + 1))
        sampled_frames = segment.get("sampled_frames", [])

        frame_results = []
        class_counter = Counter()
        frames_with_target = 0
        total_detections = 0
        max_score = 0.0
        num_missing_frames = 0
        valid_frame_candidates: List[Dict] = []
        target_frame_candidates: List[Dict] = []

        for item in sampled_frames:
            frame_path = resolve_frame_path(item)
            timestamp_sec = item.get("timestamp_sec")
            frame_index = item.get("frame_index")
            rank = item.get("rank")

            if frame_path is None:
                num_missing_frames += 1
                frame_results.append(
                    {
                        "rank": rank,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "frame_path": None,
                        "has_target": False,
                        "num_detections": 0,
                        "detected_classes": [],
                        "max_score": 0.0,
                        "error": "frame_not_found",
                    }
                )
                continue

            detections = detector.detect_image_path(frame_path, target_classes=target_classes)
            has_target = len(detections) > 0
            if has_target:
                frames_with_target += 1
                for d in detections:
                    class_counter[d["class_name"]] += 1
                target_frame_candidates.append(
                    {
                        "frame_path": frame_path,
                        "detections": detections,
                        "max_score": max((float(d["score"]) for d in detections), default=0.0),
                        "num_detections": len(detections),
                    }
                )

            valid_frame_candidates.append(
                {
                    "frame_path": frame_path,
                    "detections": detections,
                    "max_score": max((float(d["score"]) for d in detections), default=0.0),
                    "num_detections": len(detections),
                }
            )

            total_detections += len(detections)
            if detections:
                max_score = max(max_score, max(float(d["score"]) for d in detections))

            frame_results.append(
                {
                    "rank": rank,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "frame_path": frame_path,
                    "has_target": has_target,
                    "num_detections": len(detections),
                    "detected_classes": sorted({d["class_name"] for d in detections}),
                    "max_score": max((float(d["score"]) for d in detections), default=0.0),
                }
            )

        has_target_scene = frames_with_target > 0
        if has_target_scene:
            num_scenes_with_target += 1

        scene_image_name = f"{scene_image_prefix}_{seg_id:03d}.jpg"
        scene_image_path = os.path.join(video_dir, scene_image_name)
        scene_image_saved = False
        scene_image_error = None
        chosen_frame_path = None

        chosen = None
        if target_frame_candidates:
            chosen = max(
                target_frame_candidates,
                key=lambda x: (float(x["max_score"]), int(x["num_detections"])),
            )
        elif valid_frame_candidates:
            chosen = valid_frame_candidates[0]

        if chosen is not None:
            chosen_frame_path = str(chosen["frame_path"])
            img = cv2.imread(chosen_frame_path, cv2.IMREAD_COLOR)
            if img is None:
                scene_image_error = f"read_failed:{chosen_frame_path}"
            else:
                if chosen["detections"]:
                    img = draw_detections(img, chosen["detections"])
                ok = cv2.imwrite(scene_image_path, img)
                if ok:
                    scene_image_saved = True
                else:
                    scene_image_error = f"write_failed:{scene_image_path}"
        else:
            scene_image_error = "no_valid_frame"
            scene_image_path = None

        scene_results.append(
            {
                "segment_id": seg_id,
                "segment_start_index": segment.get("segment_start_index"),
                "segment_end_index": segment.get("segment_end_index"),
                "segment_num_frames": segment.get("segment_num_frames"),
                "num_key_frames": len(sampled_frames),
                "num_missing_key_frames": num_missing_frames,
                "num_key_frames_with_target": frames_with_target,
                "num_key_frames_without_target": max(0, len(sampled_frames) - frames_with_target - num_missing_frames),
                "has_target": has_target_scene,
                "num_detections": total_detections,
                "max_score": float(max_score),
                "detected_classes": sorted(class_counter.keys()),
                "detected_class_counts": dict(sorted(class_counter.items(), key=lambda x: x[0])),
                "scene_image_path": scene_image_path,
                "scene_image_saved": bool(scene_image_saved),
                "scene_image_source_frame": chosen_frame_path,
                "scene_image_error": scene_image_error,
                "key_frames": frame_results,
            }
        )

    out = {
        "video_dir": video_dir,
        "source_key_frame_json": key_frame_json_path,
        "model_weights": detector.weights,
        "conf_thres": detector.conf_thres,
        "iou_thres": detector.iou_thres,
        "img_size": detector.imgsz,
        "target_classes": sorted(target_classes) if target_classes else None,
        "num_scenes": len(scene_results),
        "num_scenes_with_target": num_scenes_with_target,
        "num_scenes_without_target": len(scene_results) - num_scenes_with_target,
        "scenes": scene_results,
    }

    save_json(out, os.path.join(video_dir, output_json_name))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="使用目标检测模型对关键帧进行检测，生成场景级是否有目标的 JSON。"
    )
    parser.add_argument("--scene-root", type=str, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--key-frame-subdir", type=str, default=DEFAULT_KEY_FRAME_SUBDIR)
    parser.add_argument("--key-frame-json-name", type=str, default=DEFAULT_KEY_FRAME_JSON_NAME)
    parser.add_argument("--output-json-name", type=str, default=DEFAULT_OUTPUT_JSON_NAME)
    parser.add_argument("--summary-json-name", type=str, default=DEFAULT_SUMMARY_JSON_NAME)
    parser.add_argument(
        "--object-detection-code-dir",
        type=str,
        default=DEFAULT_OBJECT_DETECTION_CODE_DIR,
        help="包含 models/ 与 utils/ 的目标检测代码目录",
    )
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONF_THRES)
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_IOU_THRES)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument(
        "--target-classes",
        type=str,
        default=DEFAULT_TARGET_CLASSES,
        help="仅统计这些类别（逗号分隔，留空表示任意类别都算目标）",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help="仅处理前 N 个视频目录。0 表示处理全部。",
    )
    parser.add_argument(
        "--scene-image-prefix",
        type=str,
        default=DEFAULT_SCENE_IMAGE_PREFIX,
        help="每个场景输出图片文件名前缀，保存到视频层级目录。",
    )
    args = parser.parse_args()

    target_classes = parse_target_classes(args.target_classes)
    detector = YoloSceneDetector(
        object_detection_code_dir=args.object_detection_code_dir,
        weights=args.weights,
        device=args.device,
        imgsz=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
    )

    key_frame_json_paths = find_key_frame_jsons(
        scene_root=args.scene_root,
        key_frame_subdir=args.key_frame_subdir,
        key_frame_json_name=args.key_frame_json_name,
    )
    if args.max_videos > 0:
        key_frame_json_paths = key_frame_json_paths[: args.max_videos]

    if not key_frame_json_paths:
        print("[WARN] 未找到可处理目录（缺少 key_frame_selection.json）。")
        return

    summary_items = []
    total_scenes = 0
    total_scenes_with_target = 0

    for idx, key_frame_json_path in enumerate(key_frame_json_paths, start=1):
        print(f"[{idx}/{len(key_frame_json_paths)}] PROCESS {key_frame_json_path}")
        one = process_one_video(
            key_frame_json_path=key_frame_json_path,
            detector=detector,
            target_classes=target_classes,
            output_json_name=args.output_json_name,
            scene_image_prefix=args.scene_image_prefix,
        )
        total_scenes += int(one["num_scenes"])
        total_scenes_with_target += int(one["num_scenes_with_target"])

        summary_items.append(
            {
                "video_dir": one["video_dir"],
                "source_key_frame_json": one["source_key_frame_json"],
                "output_json": os.path.join(one["video_dir"], args.output_json_name),
                "num_scenes": one["num_scenes"],
                "num_scenes_with_target": one["num_scenes_with_target"],
                "num_scenes_without_target": one["num_scenes_without_target"],
            }
        )

    summary = {
        "scene_root": args.scene_root,
        "num_videos_processed": len(summary_items),
        "num_total_scenes": total_scenes,
        "num_total_scenes_with_target": total_scenes_with_target,
        "num_total_scenes_without_target": total_scenes - total_scenes_with_target,
        "target_classes": sorted(target_classes) if target_classes else None,
        "videos": summary_items,
    }
    summary_path = os.path.join(args.scene_root, args.summary_json_name)
    save_json(summary, summary_path)

    print(f"[DONE] videos={len(summary_items)} total_scenes={total_scenes}")
    print(f"[DONE] summary_json={summary_path}")


if __name__ == "__main__":
    main()

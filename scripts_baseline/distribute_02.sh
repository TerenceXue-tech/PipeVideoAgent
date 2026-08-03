#!/usr/bin/env bash
set -euo pipefail

# =========================
# 可配置参数（直接改这里）
# =========================
ROOT_DIR="/data/xtc/PipeVideo"
PYTHON_BIN="/home/xtc/.conda/envs/pipevideo/bin/python"
SCRIPT_PATH="${ROOT_DIR}/scripts_baseline/02_get_key_frame.py"

# scene_segmentation 输出根目录（02脚本会在其下各视频目录写 key_frames）
SCENE_ROOT="${ROOT_DIR}/scene_segmentation_outputs"
VIDEO_LIST_JSON="${ROOT_DIR}/video_list_1000.json"

# 并行进程数
NUM_PROCS=8

# 如需限制到某些 GPU，可配置；02 脚本主要是 CPU/IO，此项可保留默认
GPU_IDS_CSV="0,1,2,3"

# 传给 02_get_key_frame.py 的参数
SHADE_PRE_SECONDS="1.0"
SHADE_POST_SECONDS="4.0"
WINDOW_SIZE="16"
TARGET_FRAMES="8"
OUTPUT_SUBDIR="key_frames"
MEAN_MOTION_WEIGHT="1"
ACTIVE_RATIO_WEIGHT="none"
PEAK_WEIGHT="none"
SHARPNESS_WEIGHT="none"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="${ROOT_DIR}/tmp_keyframe_split_${RUN_TAG}"
LOG_DIR="${ROOT_DIR}/logs_keyframe_split_${RUN_TAG}"
KEEP_TMP_JSON=0
KEEP_LOG_DIR=1

mkdir -p "${TMP_DIR}" "${LOG_DIR}" "${SCENE_ROOT}"

if [[ ! -f "${VIDEO_LIST_JSON}" ]]; then
  echo "[ERROR] video list not found: ${VIDEO_LIST_JSON}"
  exit 1
fi
if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "[ERROR] scene root not found: ${SCENE_ROOT}"
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
  echo "[ERROR] no GPU ids found in GPU_IDS_CSV=${GPU_IDS_CSV}"
  exit 1
fi

echo "[INFO] SCENE_ROOT=${SCENE_ROOT}"
echo "[INFO] VIDEO_LIST_JSON=${VIDEO_LIST_JSON}"
echo "[INFO] NUM_PROCS=${NUM_PROCS}"
echo "[INFO] GPU_IDS=${GPU_IDS_CSV}"
echo "[INFO] TMP_DIR=${TMP_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] KEEP_TMP_JSON=${KEEP_TMP_JSON}"
echo "[INFO] KEEP_LOG_DIR=${KEEP_LOG_DIR}"

"${PYTHON_BIN}" - << 'PY' "${VIDEO_LIST_JSON}" "${TMP_DIR}" "${NUM_PROCS}"
import json
import os
import sys

video_list_path = sys.argv[1]
tmp_dir = sys.argv[2]
num_parts = int(sys.argv[3])

with open(video_list_path, "r", encoding="utf-8") as f:
    items = json.load(f)

if not isinstance(items, list):
    raise ValueError(f"video list must be a list, got: {type(items)}")

chunks = [[] for _ in range(num_parts)]
for idx, item in enumerate(items):
    chunks[idx % num_parts].append(item)

for part_idx, chunk in enumerate(chunks):
    out_path = os.path.join(tmp_dir, f"video_list_part_{part_idx:02d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)

print(f"[SPLIT] total={len(items)}, parts={num_parts}")
PY

PIDS=()
STARTED=0
CLEANED_UP=0

cleanup_tmp() {
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ "${KEEP_TMP_JSON}" -eq 0 ]]; then
    rm -rf "${TMP_DIR}"
    echo "[CLEAN] removed split json dir: ${TMP_DIR}"
  else
    echo "[CLEAN] keep split json dir: ${TMP_DIR}"
  fi

  if [[ "${KEEP_LOG_DIR}" -eq 0 ]]; then
    rm -rf "${LOG_DIR}"
    echo "[CLEAN] removed log dir: ${LOG_DIR}"
  else
    echo "[CLEAN] keep log dir: ${LOG_DIR}"
  fi
}

terminate_children() {
  local live_pids=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      live_pids+=("${pid}")
    fi
  done

  if [[ "${#live_pids[@]}" -eq 0 ]]; then
    return
  fi

  echo "[CANCEL] terminating ${#live_pids[@]} child processes..."
  kill "${live_pids[@]}" 2>/dev/null || true
  sleep 1

  for pid in "${live_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}

on_signal() {
  local sig="${1}"
  echo "[CANCEL] received ${sig}, stopping running workers..."
  terminate_children
  exit 130
}

trap cleanup_tmp EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

for ((i=0; i<NUM_PROCS; i++)); do
  SHARD_JSON="${TMP_DIR}/video_list_part_$(printf "%02d" "${i}").json"
  LOG_PATH="${LOG_DIR}/part_$(printf "%02d" "${i}").log"
  SHARD_SUMMARY="${LOG_DIR}/key_frame_selection_summary_part_$(printf "%02d" "${i}").json"
  GPU_ID="${GPU_IDS[$((i % GPU_COUNT))]}"

  if [[ ! -f "${SHARD_JSON}" ]]; then
    echo "[WARN] shard missing, skip: ${SHARD_JSON}"
    continue
  fi

  SHARD_SIZE="$("${PYTHON_BIN}" - << 'PY' "${SHARD_JSON}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
PY
)"

  if [[ "${SHARD_SIZE}" -eq 0 ]]; then
    echo "[INFO] skip empty shard: ${SHARD_JSON}"
    continue
  fi

  echo "[LAUNCH] part=${i} size=${SHARD_SIZE} gpu=${GPU_ID} log=${LOG_PATH}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --scene-root "${SCENE_ROOT}" \
    --video-list "${SHARD_JSON}" \
    --shade-pre-seconds "${SHADE_PRE_SECONDS}" \
    --shade-post-seconds "${SHADE_POST_SECONDS}" \
    --window-size "${WINDOW_SIZE}" \
    --target-frames "${TARGET_FRAMES}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --mean-motion-weight "${MEAN_MOTION_WEIGHT}" \
    --active-ratio-weight "${ACTIVE_RATIO_WEIGHT}" \
    --peak-weight "${PEAK_WEIGHT}" \
    --sharpness-weight "${SHARPNESS_WEIGHT}" \
    --summary-path "${SHARD_SUMMARY}" \
    > "${LOG_PATH}" 2>&1 &

  PIDS+=("$!")
  STARTED=$((STARTED + 1))
done

if [[ "${STARTED}" -eq 0 ]]; then
  echo "[ERROR] no process started"
  exit 1
fi

echo "[INFO] started ${STARTED} processes, waiting..."

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

cleanup_tmp
trap - EXIT

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[DONE] finished with failures, check logs: ${LOG_DIR}"
  exit 1
fi

echo "[DONE] all processes completed successfully"
echo "[INFO] logs dir: ${LOG_DIR}"

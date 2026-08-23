# PipeVideoAgent

<p align="center">
  <b>A Video-LLM-Based Agent for Long-Distance Oil and Gas Pipeline Inspection</b>
</p>

<p align="center">
  <a href="https://github.com/TerenceXue-tech/PipeVideoAgent">
    <img src="https://img.shields.io/badge/GitHub-PipeVideoAgent-181717?logo=github" alt="GitHub">
  </a>
  <a href="https://modelscope.cn/datasets/TerenceXue/PipeVideoAgent">
    <img src="https://img.shields.io/badge/ModelScope-Dataset-624AFF" alt="Dataset">
  </a>
  <a href="https://modelscope.cn/models/TerenceXue/PipeVideoLM">
    <img src="https://img.shields.io/badge/ModelScope-PipeVideoLM-624AFF" alt="Model">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License">
  </a>
</p>

## Overview

**PipeVideoAgent** is a hierarchical video-analysis framework for risk assessment in long-distance oil and gas pipeline inspection.

Conventional image-based detectors can identify people, vehicles, and construction machinery in individual frames, but they cannot reliably distinguish static presence from motion, approach, departure, excavation, towing, or other temporally defined activities. PipeVideoAgent addresses this limitation by combining:

1. **SMoKE** — scene-wise motion-window key-frame extraction;
2. **PipeVideoLM** — a domain-adapted scene-level Video-LLM;
3. **Cross-scene semantic fusion** — traceable video-level risk aggregation.

PipeVideoAgent denotes the complete video-level pipeline. PipeVideoLM is its contained scene-understanding model rather than a parallel system.

## Resources

| Resource | Description | Link |
|---|---|---|
| Code | SMoKE scripts, SFT data, and target-detection module | [GitHub](https://github.com/TerenceXue-tech/PipeVideoAgent) |
| Dataset | Raw inspection videos and materialized scene-processing outputs | [ModelScope Dataset](https://modelscope.cn/datasets/TerenceXue/PipeVideoAgent) |
| Best model | Standalone merged PipeVideoLM based on Qwen3.5-9B | [ModelScope Model](https://modelscope.cn/models/TerenceXue/PipeVideoLM) |

## Method

### SMoKE

SMoKE, short for **scene-wise motion-window key-frame extraction**, converts a raw PTZ inspection video into compact scene-organized key-frame sequences.

It includes:

- adaptive stable-scene segmentation using grayscale-histogram changes;
- removal of complete PTZ viewpoint-transition intervals;
- SigLIP2-based semantic-motion analysis;
- contiguous motion-window selection;
- adjacent-scene redundancy suppression;
- optional deployment-time filtering of target-free scenes.

Unlike sparse global sampling, SMoKE retains temporally contiguous evidence that supports motion and behavior understanding.

### PipeVideoLM

PipeVideoLM adapts **Qwen3.5-9B** to pipeline inspection using LoRA. Given the chronologically ordered key frames of one stable scene, it predicts structured semantics including:

- target objects and locations;
- pipeline proximity;
- motion trends;
- behavioral events;
- on-site control status;
- scene-level risk.

The released ModelScope checkpoint is a standalone merged model, so a separate base model or LoRA adapter is not required for inference.

### Cross-Scene Semantic Fusion

For a video containing multiple stable scenes, PipeVideoAgent:

1. orders the scene predictions chronologically;
2. computes the video risk using a deterministic maximum-risk rule;
3. retains all scenes supporting the maximum-risk decision;
4. organizes the scene semantics into a coherent video-level summary;
5. prevents the language-model fusion stage from modifying locked safety-critical fields.

This produces a traceable path from selected key frames to scene-level evidence and the final video-level conclusion.

## Repository Structure

```text
PipeVideoAgent/
├── scripts_SMoKE/
│   ├── 00_make_video_list.py
│   ├── 01_scene_segmentation.py
│   ├── 02_get_key_frame.py
│   ├── 03_refine_scene.py
│   ├── 04_detect_scene_targets.py
│   ├── distribute_01.sh
│   ├── distribute_02.sh
│   └── distribute_03.sh
├── sft_data/
│   ├── qwen3_5_sft_train.json
│   └── qwen3_5_sft_test.json
├── object_detection/
│   ├── code/
│   │   ├── model/best_day.pt
│   │   ├── models/
│   │   └── utils/
│   ├── LICENSE
│   └── NOTICE.md
├── LICENSE
└── README.md
```

The current GitHub release focuses on SMoKE preprocessing, scene-level SFT data, and the custom target detector. Large videos, materialized preprocessing outputs, and the merged PipeVideoLM weights are hosted separately on ModelScope.

## Main Results

The following results are reported in the accompanying manuscript.

| Component | Metric | Result |
|---|---|---:|
| SMoKE | Exact scene-count accuracy | **92.16%** |
| SMoKE | Scene-boundary F1@1s | **82.93%** |
| SMoKE | Scene-boundary OSPA | **17.29%** |
| SMoKE | Average selected frames per video | **12.31** |
| SMoKE | Selector runtime | **4.86 ± 1.12 s/video** |
| PipeVideoLM | Scene-level risk accuracy | **84.17%** |
| PipeVideoLM | Scene-level risk macro-F1 | **84.41%** |
| PipeVideoAgent | Video-level risk accuracy | **84.32%** |
| PipeVideoAgent | Video-level risk macro-F1 | **84.77%** |
| PipeVideoAgent | High-risk F1 | **88.52%** |
| PipeVideoAgent | Mean semantic F1 | **82.01%** |

The selector-runtime benchmark was measured on four NVIDIA RTX 5880 Ada GPUs. It measures processing from the raw video path to materialized selected frames and should not be interpreted as end-to-end PipeVideoLM latency.

## Dataset

The ModelScope dataset contains raw pipeline inspection videos and materialized scene-processing outputs:

- [`video_data/`](https://modelscope.cn/datasets/TerenceXue/PipeVideoAgent)
- [`scene_segmentation_outputs_v2/`](https://modelscope.cn/datasets/TerenceXue/PipeVideoAgent)

The paper uses video-level splitting, ensuring that scenes from the same source video do not appear in different subsets.

| Experimental set | Samples | Low | Medium | High |
|---|---:|---:|---:|---:|
| Scene-level training | 908 scenes | 417 | 378 | 113 |
| Scene-level test | 398 scenes | 196 | 161 | 41 |
| Video-level training | 548 videos | 161 | 285 | 102 |
| Video-level test | 236 videos | 81 | 123 | 32 |

### Download the Dataset

Install the ModelScope client:

```bash
pip install -U modelscope
```

Download the dataset:

```bash
modelscope download \
  --dataset TerenceXue/PipeVideoAgent \
  --local_dir ./data/PipeVideoAgent
```

Because the dataset contains video files and materialized scene outputs, verify the required storage space before downloading the complete repository.

## Installation

### 1. Clone the Repository

Git LFS is required for the detector weights and font files.

```bash
git lfs install
git clone https://github.com/TerenceXue-tech/PipeVideoAgent.git
cd PipeVideoAgent
git lfs pull
```

### 2. Create an Environment

```bash
conda create -n pipevideoagent python=3.10 -y
conda activate pipevideoagent
```

Install the main dependencies:

```bash
pip install \
  torch torchvision \
  "transformers>=5.12.1" \
  accelerate safetensors \
  modelscope \
  opencv-python pillow numpy matplotlib \
  openpyxl scipy pandas pyyaml tqdm requests \
  seaborn flask psutil
```

CUDA-enabled PyTorch should be installed according to the CUDA version of the target machine.

A CUDA GPU is strongly recommended for SigLIP2 feature extraction, target detection, and PipeVideoLM inference.

## Running SMoKE

### 1. Prepare a Video List

SMoKE expects a JSON list containing absolute video paths:

```json
[
  {
    "video_path": "/absolute/path/to/video_001.mp4"
  },
  {
    "video_path": "/absolute/path/to/video_002.mp4"
  }
]
```

Alternatively, generate the list from the first column of an Excel file:

```bash
python scripts_SMoKE/00_make_video_list.py \
  --excel-path /path/to/labeled_data.xlsx \
  --video-data-dir /path/to/video_data \
  --output /path/to/video_list.json
```

### 2. Adaptive Stable-Scene Segmentation

Set the local SigLIP2 checkpoint, input list, and output directory:

```bash
export MODEL_PATH=/path/to/siglip2-giant-opt-patch16-384
export VIDEO_LIST_PATH=/path/to/video_list.json
export OUTPUT_DIR=/path/to/scene_segmentation_outputs

python scripts_SMoKE/01_scene_segmentation.py
```

For exact alignment with the paper configuration, set the following constants in `scripts_SMoKE/01_scene_segmentation.py` before running:

```python
SAMPLE_FPS = 4
GRAY_HIST_BINS = 1536
THRESHOLD_STD_FACTOR = 2.4
MIN_PEAK_GAP_SECONDS = 6.0
```

The currently released script may contain development defaults for the last two values; use the values above when reproducing the reported experiments.

### 3. Motion-Window Key-Frame Selection

```bash
python scripts_SMoKE/02_get_key_frame.py \
  --scene-root "$OUTPUT_DIR" \
  --video-list "$VIDEO_LIST_PATH" \
  --window-size 16 \
  --target-frames 8 \
  --mean-motion-weight 0.7 \
  --active-ratio-weight 0.1 \
  --peak-weight 0.1 \
  --sharpness-weight 0.1
```

The paper uses:

| Parameter | Value |
|---|---:|
| Sampling rate | 4 frames/s |
| Window length | 16 sampled frames |
| Window stride | 1 |
| Active-motion threshold | 0.08 |
| Mean-motion weight | 0.7 |
| Persistence weight | 0.1 |
| Peak-motion weight | 0.1 |
| Sharpness weight | 0.1 |
| Output key frames per scene | 8 |

### 4. Adjacent-Scene Deduplication

A dry run can be used to inspect the planned changes:

```bash
python scripts_SMoKE/03_refine_scene.py \
  --scene-root "$OUTPUT_DIR" \
  --video-list "$VIDEO_LIST_PATH" \
  --similarity-threshold 0.97 \
  --dry-run
```

Apply the refinement:

```bash
python scripts_SMoKE/03_refine_scene.py \
  --scene-root "$OUTPUT_DIR" \
  --video-list "$VIDEO_LIST_PATH" \
  --similarity-threshold 0.97
```

### 5. Deployment-Time Target Filtering

```bash
python scripts_SMoKE/04_detect_scene_targets.py \
  --scene-root "$OUTPUT_DIR" \
  --object-detection-code-dir "$(pwd)/object_detection/code" \
  --weights "$(pwd)/object_detection/code/model/best_day.pt" \
  --device cuda:0 \
  --img-size 960 \
  --conf-thres 0.4 \
  --iou-thres 0.4 \
  --target-classes "loader,excavator,person,car,truck,tanker,dumb_truck"
```

> **Important:** target filtering is a deployment-time computation-reduction stage. It was disabled when constructing the training, validation, and evaluation subsets reported in the paper.

## Downloading and Loading PipeVideoLM

Download the standalone merged model:

```bash
modelscope download \
  --model TerenceXue/PipeVideoLM \
  --local_dir ./models/PipeVideoLM
```

Load it using Transformers:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_dir = "./models/PipeVideoLM"

processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()
```

You can also download it programmatically:

```python
from modelscope import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

model_dir = snapshot_download("TerenceXue/PipeVideoLM")

processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForImageTextToText.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)
```

For scene-level inference, provide the selected key frames of one stable scene in chronological order. The task instruction should request the structured object, location, proximity, motion, behavior, control-status, and risk fields defined by PipeVideoLM.

## Paper-Aligned Configuration

The principal SMoKE configuration reported in the paper is:

| Symbol or option | Value |
|---|---:|
| Sampling rate | 4 frames/s |
| Grayscale histogram bins | 1536 |
| Adaptive threshold coefficient | 2.4 |
| Minimum boundary gap | 6.0 s |
| SigLIP2 checkpoint | `siglip2-giant-opt-patch16-384` |
| Motion-window length and stride | `(16, 1)` |
| Active-motion threshold | 0.08 |
| Window weights | `(0.7, 0.1, 0.1, 0.1)` |
| Selected frames per scene | 8 |
| Adjacent-scene similarity threshold | 0.97 |
| Detector image size | 960 |
| Detector confidence threshold | 0.4 |
| Detector IoU threshold | 0.4 |

## Reproduction Notes

- Keep all scenes from the same source video in the same data subset.
- Do not apply deployment-time target filtering to training, validation, or evaluation subsets.
- Preserve the chronological order of selected key frames.
- Preserve scene identifiers during cross-scene fusion so that every conclusion remains traceable to its supporting scenes.
- Average selected-frame count measures downstream visual-input and prompt-prefill load; it is not, by itself, an end-to-end speed measurement.
- The hosted PipeVideoLM checkpoint was selected post hoc using corrected-test risk accuracy. Its reported 84.17% score should therefore be treated as the released checkpoint-selection result; a new held-out set is recommended for an unbiased deployment estimate.

## Limitations and Responsible Use

PipeVideoAgent is intended for research and decision support in industrial video inspection. It is not a replacement for certified safety procedures or human review.

Performance may degrade under:

- severe occlusion;
- low illumination;
- very small or distant targets;
- unseen camera devices or environments;
- activities outside the training-domain risk definitions.

High-risk predictions and automated filtering decisions should be reviewed by qualified operators before operational action is taken.

## License

The repository-level code is released under the [Apache License 2.0](LICENSE).

The `object_detection/` subtree contains YOLOv5-derived components that retain their original AGPL-3.0 licensing and notices. The repository-level Apache-2.0 license does not relicense those third-party components.

The dataset and model are additionally subject to the license terms displayed on their respective ModelScope pages.

## Citation

If this project is useful in your research, please cite:

```bibtex
@misc{xue2026pipevideoagent,
  title        = {PipeVideoAgent: A Video-LLM-Based Agent for Long-Distance Oil and Gas Pipeline Inspection},
  author       = {Xue, Tianci and Liu, Zhaoxu and Shang, Chao and Cheng, Zhipeng and Chen, Yong and Ma, Ji and Guo, Yong},
  year         = {2026},
  howpublished = {Manuscript}
}
```

import time
import os
from flask import Flask, request, jsonify
import numpy as np
import cv2
import torch
from utils.augmentations import  letterbox
import torch.nn.functional as F
from utils.general import non_max_suppression
from utils.torch_utils import select_device
# from models.yolo import Detect
from models.experimental import attempt_load
import platform
import pathlib
from pathlib import Path
import io
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath


app = Flask(__name__)
# 初始化YOLO模型
device = select_device("0")
half = device.type != 'cpu'
print(half)
yolo_model_b = attempt_load(ROOT/'model/best_day.pt')
yolo_model_b.to(device).eval()
yolo_model_b_class_names = yolo_model_b.module.names if hasattr(yolo_model_b, 'module') else yolo_model_b.names
if half:
    yolo_model_b.half()

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
@app.route('/predict', methods=["GET", "POST"])
def predict():
    # 获取图片
    try:
        time0 = time.time()
        file = request.files.get('file')
        results = []
        name_class_dict = {"loader": "铲车",
                           "fire": "火焰",
                           "excavator": "挖掘机",
                           "person": "行人",
                           "car": "汽车",
                           "truck": "货车",
                           "tanker": "罐车",
                           "dumb_truck": "重车",
                           }
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img0 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = letterbox(img0, 960, stride=64, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        half = True
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = yolo_model_b(img, augment=False)[0]  # 0.22s
        pred = pred.float()
        pred = non_max_suppression(pred, 0.4, 0.4)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy = [int(x) for x in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    score = round(conf.tolist(), 2)
                    lbl_zw = yolo_model_b_class_names[int(cls)]
                    if score >= conf_tmp:
                        results.append({
                            'className': name_class_dict[lbl_zw],
                            'score': float(score),
                            'box': xyxy,
                        })
    except Exception as e:
        print("识别异常**************")
        results = []
    print(results)
    time1 = time.time()
    cost_time =time1 - time0
    print("cost_time: ", cost_time)
    return jsonify({'predictions': results, 'success': True}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True, debug=False, port=9515)
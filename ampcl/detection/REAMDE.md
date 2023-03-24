# README

## YOLOv5

- 安装相关依赖


```bash
$ git clone https://github.com/ultralytics/yolov5.git
$ cd yolov5
$ pip3 install -r requirements.txt
```

- Usage

```python
from ampcl.detection import SimpleYolo
import cv2

simple_yolo = SimpleYolo()
img = cv2.imread("图片路径")
# 只保留一部分的检测结果（idx的映射详看"simple_yolov5.py"）
results = simple_yolo.infer(img, keep_idx=[0, 1, 2, 3, 5, 7])

# input: image
# output: box2d [N, 6] (x1, y1, x2, y2, score, cls_id)
```
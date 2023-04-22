import numpy as np
import torch

cls_id_map = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
              8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
              22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
              29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
              35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
              40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
              48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
              55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
              62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
              69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
              76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


class SimpleYolo:
    def __init__(self):
        # or yolov5n, yolov5x6, custom, etc.
        # note：需联网
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5x")
        self.cls_id_map = cls_id_map

    def infer(self, img, keep_idx=None, use_conf=False):
        """
        :param img:
        :param list keep_idx: 只保留特定类别的box
        :return: [N, 7] [x1, y1, x2, y2, conf, cls_id]
        """
        results = self.model(img)
        box2d_num = len(results.pred[0])
        box2d = []
        for i in range(box2d_num):
            cls = results.pred[0][:, 5].cpu().numpy()
            if (keep_idx is not None) and (int(cls[i]) not in keep_idx):
                continue
            xyxy = results.pred[0][:, :4].cpu().numpy()
            conf = results.pred[0][:, 4].cpu().numpy()
            box2d.append(np.hstack((xyxy[i], cls[i], conf[i])))
        box2d = np.asarray(box2d)
        return box2d

    def infer_and_remap(self, img, keep_idx=None):
        box2d = self.infer(img, keep_idx)
        id_remap_dict = {
            0: 2,  # person -> Pedestrian
            1: 3,  # bicycle -> Cyclist
            2: 1,  # car->Car
            3: 3,  # motorcycle->Cyclist
            5: 1,  # bus -> Car
            7: 1  # truck -> Car
        }
        for i in range(box2d.shape[0]):
            box2d[i, 4] = id_remap_dict[box2d[i, 4]]
        return box2d

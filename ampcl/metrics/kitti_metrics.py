import warnings

import numba
import numpy as np
from numba.core.errors import NumbaPerformanceWarning

from ..calibration import object3d
from .rotate_iou import rotate_iou_gpu_eval

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


class Statistics(object):
    def __init__(self, name, num):
        self.name = name
        self.num = num
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __repr__(self):
        return f'{self.name} num: {self.num}, tp: {self.tp}, fp: {self.fp}, fn: {self.fn}'


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def calculate_overlap_wrapper(gt_box, gt_cls_id, pred_box, pred_cls_id, overlap_func, threshold=0.5):
    """
    :param gt_box: (N, 4) or (N, 7)
    :param pred_box: (M, 4) or (M, 7)
    :return: (N, M)
    """
    gt_num = gt_box.shape[0]
    pred_num = pred_box.shape[0]
    overlap = overlap_func(gt_box, pred_box)
    is_used = np.zeros((pred_num,), dtype=np.bool_)
    pred_iou_values = np.full((pred_num,), -1.0, dtype=np.float32)

    # 给每个真值找对应的检测框
    for i in range(gt_num):
        for j in range(pred_num):
            # detection is assigned to a gt, and no more overlap computation is needed
            if is_used[j]:
                continue
            # the class of detection and gt should be the same
            if gt_cls_id[i] != pred_cls_id[j]:
                continue
            if overlap[i, j] > threshold:
                is_used[j] = 1
                pred_iou_values[j] = overlap[i, j]

    return pred_iou_values


def calculate_img_box_overlap(gt_box2d4c, pred_box2d4c, threshold=0.5):
    """
    :param gt_box2d4c: (N, 4)
    :param pred_box2d4c: (M, 4)
    :return: (N, M)
    """
    return calculate_overlap_wrapper(gt_box2d4c, pred_box2d4c, image_box_overlap, threshold=threshold)


def calculate_bev_box_overlap(gt_box3d_cam, pred_box3d_cam, threshold=0.5):
    """
    :param gt_box2d4c: (N, 7) (x, y, z, l, h, w, ry)
    :param pred_box2d4c: (M, 7) (x, y, z, l, h, w, ry)
    :return: (N, M)
    """
    return calculate_overlap_wrapper(gt_box3d_cam, pred_box3d_cam, bev_box_overlap, threshold=threshold)


def calculate_box3d_cam_overlap(gt_box3d_cam, pred_box3d_cam, threshold=0.5, criterion=-1):
    """
    :param gt_box2d4c: (N, 7) (x, y, z, l, h, w, ry)
    :param pred_box2d4c: (M, 7) (x, y, z, l, h, w, ry)
    :return: (N, M)
    """
    gt_cls_id = gt_box3d_cam[:, 7]
    pred_cls_id = pred_box3d_cam[:, 7]

    overlap = rotate_iou_gpu_eval(gt_box3d_cam[:, [0, 2, 3, 5, 6]],
                                  pred_box3d_cam[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(gt_box3d_cam, pred_box3d_cam, overlap, criterion)

    gt_num = gt_box3d_cam.shape[0]
    pred_num = pred_box3d_cam.shape[0]
    is_used = np.zeros((pred_num,), dtype=np.bool_)
    pred_iou_values = np.full((pred_num,), -1.0, dtype=np.float32)

    # 给每个真值找对应的检测框
    for i in range(gt_num):
        for j in range(pred_num):
            # detection is assigned to a gt, and no more overlap computation is needed
            if is_used[j]:
                continue
            # the class of detection and gt should be the same
            if gt_cls_id[i] != pred_cls_id[j]:
                continue
            if overlap[i, j] > threshold:
                is_used[j] = 1
                pred_iou_values[j] = overlap[i, j]

    return pred_iou_values


def calculate_box3d_lidar_overlap(gt_box3d_lidar, pred_box3d_lidar, cal_info, threshold=0.5, criterion=-1):
    """
    :param gt_box2d4c: (N, 7) (x, y, z, l, h, w, ry)
    :param pred_box2d4c: (M, 7) (x, y, z, l, h, w, ry)
    :return: (N, M)
    """
    gt_box3d_cam = object3d.box3d_lidar_to_cam(gt_box3d_lidar, cal_info)
    pred_box3d_cam = object3d.box3d_lidar_to_cam(pred_box3d_lidar, cal_info)
    return calculate_box3d_cam_overlap(gt_box3d_cam, pred_box3d_cam, threshold, criterion)


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0

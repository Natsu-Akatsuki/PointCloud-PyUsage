import pickle
from pathlib import Path

import numpy as np
from ampcl.calibration.calibration_livox import LivoxCalibration
from ampcl.calibration.object3d_kitti import get_objects_from_label
from open3d_vis_utils import draw_scenes
import common_utils


def vis_pcdet_kitti_result():
    file_path = 'data/000000.bin'
    file_path = Path(file_path).resolve()
    pointcloud = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)[:, 0:3]
    gt_infos_path = Path('data/kitti_infos_train.pkl').resolve()
    with open(str(gt_infos_path), 'rb') as f:
        gt_infos = pickle.load(f)
    gt_boxes = gt_infos[0]["annos"]["gt_boxes_lidar"]
    draw_scenes(pointcloud, gt_boxes=gt_boxes)


def vis_kitti_label():
    gt_boxes_lidar, pointcloud = generate_pcd_info()

    draw_scenes(pointcloud, gt_boxes=gt_boxes_lidar)


def generate_pcd_info():
    file_path = '/home/helios/mnt/dataset/livox_dataset/training/lidar/000000.bin'
    cal_path = "/home/helios/mnt/dataset/livox_dataset/training/calib/config.txt"
    label_path = "/home/helios/Github/workspace/livox_detection/0.txt"

    file_path = Path(file_path).resolve()
    pointcloud = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)[:, 0:3]
    label_path = Path(label_path).resolve()
    cal_path = Path(cal_path).resolve()
    obj_list = get_objects_from_label(label_path)
    calib = LivoxCalibration(cal_path)

    annotations = {}
    annotations['name'] = np.array([obj.cls_type for obj in obj_list])
    annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
    annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
    annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
    annotations['score'] = np.array([obj.score for obj in obj_list])
    annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    loc = annotations['location'][:num_objects]  # x,y,z
    dims = annotations['dimensions'][:num_objects]  # l,w,h
    rots = annotations['rotation_y'][:num_objects]  # yaw
    loc_lidar = calib.camera_to_lidar_points(loc)  # points from camera frame->lidar frame

    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2  # btn center->geometry center
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])],
                                    axis=1)
    return gt_boxes_lidar, pointcloud


def remove_foreground_point():
    bbxes, pointcloud = generate_pcd_info()
    mask = np.zeros(pointcloud.shape[0], dtype=np.bool_)
    for i in range(bbxes.shape[0]):
        box = bbxes[i]
        indices_points_inside = common_utils.get_indices_of_points_inside(pointcloud, box, margin=0.1)
        mask[indices_points_inside] = True

    pointcloud = pointcloud[mask]
    draw_scenes(pointcloud[:, :3], gt_boxes=bbxes)


if __name__ == '__main__':
    remove_foreground_point()

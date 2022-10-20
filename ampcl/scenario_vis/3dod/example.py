import pickle
from pathlib import Path

import numpy as np
from dataset_utils.calibration_kitti import Calibration
from dataset_utils.object3d_kitti import get_objects_from_label
from open3d_vis_utils import draw_scenes


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
    file_path = 'data/velodyne/000000.bin'
    cal_path = 'data/calib/000000.txt'
    label_path = 'data/label_2/000000.txt'

    file_path = Path(file_path).resolve()
    pointcloud = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)[:, 0:3]
    label_path = Path(label_path).resolve()
    cal_path = Path(cal_path).resolve()
    obj_list = get_objects_from_label(label_path)
    calib = Calibration(cal_path)

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
    loc_lidar = calib.rect_to_lidar(loc)  # points from camera frame->lidar frame

    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2  # btn center->geometry center
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])],
                                    axis=1)
    return gt_boxes_lidar, pointcloud


def remove_foreground_point():
    import numpy as np
    import torch
    from ops.roiaware_pool3d import roiaware_pool3d_utils

    gt_boxes_lidar, pointcloud = generate_pcd_info()

    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
        torch.from_numpy(pointcloud[:, 0:3]).unsqueeze(dim=0).float().cuda(),
        torch.from_numpy(gt_boxes_lidar[:, 0:7]).unsqueeze(dim=0).float().cuda()
    ).long().squeeze(dim=0).cpu().numpy()

    mask = np.ones(pointcloud.shape[0], dtype=np.bool_)
    for i in range(gt_boxes_lidar.shape[0]):
        mask[box_idxs_of_pts == i] = False
    pointcloud = pointcloud[mask]

    draw_scenes(pointcloud[:, :3])


if __name__ == '__main__':
    remove_foreground_point()

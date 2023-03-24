from . import calibration_template
import numpy as np


def box3d_from_kitti_to_lidar(box3d_kitti, cal_info):
    """
    (h, w, l, x, y, z, ry, ...) -> (x, y, z, l, w, h, rz, ...)
    :param box3d_kitti:
    :param cal_info:
    :return:
    """
    h = box3d_kitti[:, 0].reshape(-1, 1)
    w = box3d_kitti[:, 1].reshape(-1, 1)
    l = box3d_kitti[:, 2].reshape(-1, 1)
    loc = box3d_kitti[:, 3:6]
    rots = box3d_kitti[:, 6].reshape(-1, 1)
    cls_id = box3d_kitti[:, 7].reshape(-1, 1)
    loc_lidar = calibration_template.camera_to_lidar_points(loc, cal_info)
    loc_lidar[:, 2] += h[:, 0] / 2
    if box3d_kitti.shape[1] > 7:
        extra_info_length = box3d_kitti.shape[1] - 7
        extra_info = box3d_kitti[:, 7:].reshape(-1, extra_info_length)
    else:
        extra_info = np.zeros(0, 7)
    box3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots), cls_id, extra_info], axis=1)
    return box3d_lidar


def box3d_from_lidar_to_kitti(box3d_lidar, cal_info):
    """
    (x, y, z, l, w, h, rz) -> (h, w, l, x, y, z, ry)
    :param box3d_lidar:
    :param cal_info:
    :return:
    """
    loc = box3d_lidar[:, :3]

    h = box3d_lidar[:, 5].reshape(-1, 1)
    w = box3d_lidar[:, 4].reshape(-1, 1)
    l = box3d_lidar[:, 3].reshape(-1, 1)

    rots = box3d_lidar[:, 6].reshape(-1, 1)
    cls_id = box3d_lidar[:, 7].reshape(-1, 1)
    loc_camera = calibration_template.lidar_to_camera_points(loc, cal_info)
    loc_camera[:, 2] -= h[:, 0] / 2

    if box3d_lidar.shape[1] > 7:
        extra_info_length = box3d_lidar.shape[1] - 7
        extra_info = box3d_lidar[:, 7:].reshape(-1, extra_info_length)
    else:
        extra_info = np.zeros(0, 7)

    box3d_kitti = np.concatenate([h, w, l, loc_camera, -(np.pi / 2 + rots), cls_id, extra_info], axis=1)
    return box3d_kitti

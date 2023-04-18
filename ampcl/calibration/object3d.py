import copy

from . import calibration
import numpy as np


def box3d_kitti_to_lidar(box3d_kitti, cal_info):
    """
    (h, w, l, x, y, z, ry, ...) -> (x, y, z, l, w, h, rz, ...)
    :param box3d_kitti: [N, 7] or [N, 7+extra_info]
    :param cal_info:
    :return:
    """
    h = box3d_kitti[:, 0].reshape(-1, 1)
    w = box3d_kitti[:, 1].reshape(-1, 1)
    l = box3d_kitti[:, 2].reshape(-1, 1)
    loc = box3d_kitti[:, 3:6]
    rots = box3d_kitti[:, 6].reshape(-1, 1)
    loc_lidar = calibration.camera_to_lidar_points(loc, cal_info)
    loc_lidar[:, 2] += h[:, 0] / 2
    if box3d_kitti.shape[1] > 7:
        extra_info_length = box3d_kitti.shape[1] - 7
        extra_info = box3d_kitti[:, 7:].reshape(-1, extra_info_length)
    else:
        extra_info = np.zeros(0, 7)
    box3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots), extra_info], axis=1)
    return box3d_lidar


def box3d_lidar_to_kitti(box3d_lidar, cal_info):
    """
    (x, y, z, l, w, h, rz, ...) -> (h, w, l, x, y, z, ry, ...)
    :param box3d_lidar: [N, 7] or [N, 7+extra_info]
    :param cal_info:
    :return:
    """
    loc = box3d_lidar[:, :3]

    h = box3d_lidar[:, 5].reshape(-1, 1)
    w = box3d_lidar[:, 4].reshape(-1, 1)
    l = box3d_lidar[:, 3].reshape(-1, 1)

    rots = box3d_lidar[:, 6].reshape(-1, 1)
    loc_camera = calibration.lidar_to_camera_points(loc, cal_info)
    loc_camera[:, 2] -= h[:, 0] / 2

    if box3d_lidar.shape[1] > 7:
        extra_info_length = box3d_lidar.shape[1] - 7
        extra_info = box3d_lidar[:, 7:].reshape(-1, extra_info_length)
    else:
        extra_info = np.zeros(0, 7)

    box3d_kitti = np.concatenate([h, w, l, loc_camera, -(np.pi / 2 + rots), extra_info], axis=1)
    return box3d_kitti


def box3d_lidar_to_cam(box3d_lidar, cal_info):
    """
    :param box3d_lidar: (N, 7) [x, y, z, l, w, h, rz]
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, ry] in camera frame
    """
    box3d_lidar_copy = copy.deepcopy(box3d_lidar)
    xyz_lidar = box3d_lidar_copy[:, 0:3]
    l, w, h = box3d_lidar_copy[:, 3:4], box3d_lidar_copy[:, 4:5], box3d_lidar_copy[:, 5:6]
    r = box3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calibration.lidar_to_camera_points(xyz_lidar, cal_info=cal_info)

    r = -r - np.pi / 2

    if box3d_lidar.shape[1] > 7:
        extra_info_length = box3d_lidar.shape[1] - 7
        extra_info = box3d_lidar[:, 7:].reshape(-1, extra_info_length)
    else:
        extra_info = np.zeros(0, 7)

    box3d_cam = np.concatenate([xyz_cam, l, h, w, r, extra_info], axis=-1)
    return box3d_cam


def box3d_cam_to_2d(box3d_cam, cal_info, img_shape):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """

    corners3d = box3d_cam_to_3d8c(box3d_cam.astype(np.float32))
    pts_img, _, _ = calibration.cam_to_pixel(corners3d.reshape(-1, 3), cal_info, img_shape=None)
    box2d8c = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(box2d8c, axis=1)  # (N, 2)
    max_uv = np.max(box2d8c, axis=1)  # (N, 2)
    box2d4c = np.concatenate([min_uv, max_uv], axis=1)
    if img_shape is not None:
        box2d4c[:, 0] = np.clip(box2d4c[:, 0], a_min=0, a_max=img_shape[1] - 1)
        box2d4c[:, 1] = np.clip(box2d4c[:, 1], a_min=0, a_max=img_shape[0] - 1)
        box2d4c[:, 2] = np.clip(box2d4c[:, 2], a_min=0, a_max=img_shape[1] - 1)
        box2d4c[:, 3] = np.clip(box2d4c[:, 3], a_min=0, a_max=img_shape[0] - 1)

    return box2d4c, box2d8c


def box3d_cam_to_3d8c(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def box3d_lidar_to_2d(box3d_lidar, cal_info, img_shape):
    box3d_cam = box3d_lidar_to_cam(box3d_lidar, cal_info)
    box2d4c, box2d8c = box3d_cam_to_2d(box3d_cam, cal_info, img_shape=img_shape)
    return box2d4c, box2d8c


def corner_3d8c_to_2d4c_2d8c(corners3d, cal_info, frame="lidar", img_shape=None):
    """
    :param cal_info:
    :param frame:
    :param corners3d: (8, 3) corners in lidar/camera coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8, 2) [xi, yi] in rgb coordinate
    """
    img_pts = None
    if frame == "lidar":
        img_pts, _, _ = calibration.lidar_to_pixel(corners3d, cal_info, img_shape)
    elif frame == "camera":
        img_pts, _, _ = calibration.cam_to_pixel(corners3d, cal_info, img_shape)

    x, y = img_pts[:, 0], img_pts[:, 1]
    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    box2d4c = np.stack((x1, y1, x2, y2))
    box2d8c = np.stack((x, y), axis=1)

    if img_shape is not None:
        box2d4c[0] = np.clip(box2d4c[0], a_min=0, a_max=img_shape[1] - 1)
        box2d4c[1] = np.clip(box2d4c[1], a_min=0, a_max=img_shape[0] - 1)
        box2d4c[2] = np.clip(box2d4c[2], a_min=0, a_max=img_shape[1] - 1)
        box2d4c[3] = np.clip(box2d4c[3], a_min=0, a_max=img_shape[0] - 1)

    return box2d4c, box2d8c

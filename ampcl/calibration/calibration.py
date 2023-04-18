import numpy as np


def to_hom_coordinate(pts):
    """生成齐次坐标
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom


def lidar_to_camera_points(pts_lidar, cal_info):
    """将激光雷达系的点转换到相机系下
    :param cal_info:
    :param pts_lidar: (N, 3)
    :return pts_camera: (N, 3)

    假定P_c和P_l是按列组成的矩阵，则有
    若数据按列组成矩阵则：P_c = extri_mat @ P_l
    但数据按行组成矩阵则：P_c.T = extri_mat @ P_l.T -> P_c = P_l @ extri_mat.T
    """
    extri_mat = cal_info["extri_mat"]
    pts_lidar_hom = to_hom_coordinate(pts_lidar)
    pts_camera = np.dot(pts_lidar_hom, extri_mat.T)[:, 0:3]
    return pts_camera


def camera_to_lidar_points(pts_camera, cal_info):
    """将相机系下的点转换到激光雷达系下
    :param pts_camera:
    :param cal_info:
    :return:
    """
    extri_mat = cal_info["extri_mat"]
    pts_camera_hom = to_hom_coordinate(pts_camera)
    pts_lidar = (pts_camera_hom @ np.linalg.inv(extri_mat.T))[:, 0:3]
    return pts_lidar


def cam_to_pixel(pts_camera, cal_info, img_shape=None):
    """ 将相机系下的点转换到图像坐标系下
    :param pts_camera: (N, 3)
    :return pts_img: (N, 2)
    """

    pts_camera_depth = pts_camera[:, 2]
    intri_mat = cal_info["intri_mat"]

    # 遵从《SLAM14讲》规范，先归一化再量化采样
    pts_img = (pts_camera.T / pts_camera_depth).T
    pts_img = np.dot(pts_img, intri_mat.T)[:, 0:2]  # (N, 2)

    mask = None
    if img_shape is not None:
        mask = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_shape[1]) & \
               (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_shape[0])
    # note：不添加截断，只有用于生成box2d时才需要截断
    return pts_img, pts_camera_depth, mask


def lidar_to_pixel(pts_lidar, cal_info, img_shape=None, use_mask=False):
    """
    :param float32 pts_lidar: (N, ...) [x, y, z, ...]
    :param img_shape:
    :param use_mask:
    :return float32 pts_img: (N, 2)
    """
    pts_camera = lidar_to_camera_points(pts_lidar[:, :3], cal_info)
    pts_img, pts_camera_depth, pt_in_img_mask = cam_to_pixel(pts_camera, cal_info, img_shape=img_shape)

    mask = None
    if use_mask:
        depth_mask = pts_camera_depth > 0
        mask = np.logical_and(pt_in_img_mask, depth_mask)
        pts_img = pts_img[mask]
        pts_camera_depth = pts_camera_depth[mask]
    return pts_img, pts_camera_depth, mask

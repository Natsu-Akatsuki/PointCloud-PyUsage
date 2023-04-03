import cv2
import numpy as np


class Calibration():
    def __init__(self):
        raise NotImplementedError("Please overwrite the init function")

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_camera_points(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_camera: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_camera = np.dot(pts_lidar_hom, self.extri_matrix.T)[:, 0:3]
        return pts_camera

    def camera_to_img_points(self, pts_camera, image_shape=None):
        """
        :param pts_camera: (N, 3)
        :return pts_img: (N, 2)
        """

        pts_camera_depth = pts_camera[:, 2]

        # 遵从slam14讲规范，先归一化再量化采样
        pts_img = (pts_camera.T / pts_camera_depth).T
        pts_img = np.dot(pts_img, self.intri_matrix.T)[:, 0:2]  # (N, 2)

        if image_shape is not None:
            pts_img[:, 0] = np.clip(pts_img[:, 0], a_min=0, a_max=image_shape[1] - 1)
            pts_img[:, 1] = np.clip(pts_img[:, 1], a_min=0, a_max=image_shape[0] - 1)

        return pts_img, pts_camera_depth

    def lidar_to_img(self, pts_lidar, image_shape=None):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_camera = self.lidar_to_camera_points(pts_lidar)
        pts_img, pts_camera_depth = self.camera_to_img_points(pts_camera, image_shape=image_shape)
        return pts_img, pts_camera_depth

    def corners3d_lidar_to_boxes2d_image(self, corners3d_lidar, image_shape=None):
        """
        :param corners3d: (8, 3) corners in lidar coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8, 2) [xi, yi] in rgb coordinate
        """

        img_pts, _ = self.lidar_to_img(corners3d_lidar, image_shape)

        x, y = img_pts[:, 0], img_pts[:, 1]
        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)

        boxes2d_image = np.stack((x1, y1, x2, y2))
        boxes_corner = np.stack((x, y), axis=1)

        if image_shape is not None:
            boxes2d_image[0] = np.clip(boxes2d_image[0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[1] = np.clip(boxes2d_image[1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[2] = np.clip(boxes2d_image[2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[3] = np.clip(boxes2d_image[3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image, boxes_corner

    # lidar -> camera
    def lidar_to_camera_lwh3d(self, boxes3d_lidar):
        """
        Args:
            boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            calib: {obj}
        Returns:
            boxes3d_camera: (N, 7) [x, y, z, l, w, h, r] in camera coord
        """
        xyz_lidar = boxes3d_lidar[:, 0:3]
        l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.lidar_to_camera_points(xyz_lidar)
        r = -r - np.pi / 2
        return np.concatenate([xyz_cam, l, w, h, r], axis=-1)

    # camera -> lidar
    def camera_to_lidar_points(self, pts_camera):
        """
        Args:pts_camera
            pts_camera: (N, 3)
        Returns:
            pts_lidar: (N, 3)
        """
        pts_camera_hom = self.cart_to_hom(pts_camera)
        pts_lidar = (pts_camera_hom @ np.linalg.inv(self.extri_matrix.T))[:, 0:3]
        return pts_lidar


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



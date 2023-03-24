import numpy as np
import cv2


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


def camera_to_img_points(pts_camera, cal_info, image_shape=None):
    """ 将相机系下的点转换到图像坐标系下
    :param pts_camera: (N, 3)
    :return pts_img: (N, 2)
    """

    pts_camera_depth = pts_camera[:, 2]
    intri_mat = cal_info["intri_mat"]

    # 遵从《SLAM14讲》规范，先归一化再量化采样
    pts_img = (pts_camera.T / pts_camera_depth).T
    pts_img = np.dot(pts_img, intri_mat.T)[:, 0:2]  # (N, 2)

    if image_shape is not None:
        pts_img[:, 0] = np.clip(pts_img[:, 0], a_min=0, a_max=image_shape[1] - 1)
        pts_img[:, 1] = np.clip(pts_img[:, 1], a_min=0, a_max=image_shape[0] - 1)

    return pts_img, pts_camera_depth


def lidar_to_img_points(pts_lidar, cal_info, image_shape=None):
    """
    :param pts_lidar: (N, 3)
    :return pts_img: (N, 2)
    """
    pts_camera = lidar_to_camera_points(pts_lidar, cal_info)
    pts_img, pts_camera_depth = camera_to_img_points(pts_camera, cal_info, image_shape=image_shape)
    return pts_img, pts_camera_depth


def corners_3d_to_2d(corners3d, cal_info, frame="lidar", image_shape=None):
    """
    :param frame:
    :param corners3d: (8, 3) corners in lidar/camera coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8, 2) [xi, yi] in rgb coordinate
    """
    img_pts = None
    if frame == "lidar":
        img_pts, _ = lidar_to_img_points(corners3d, cal_info, image_shape)
    elif frame == "camera":
        img_pts, _ = camera_to_img_points(corners3d, cal_info, image_shape, cal_info)

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


def paint_pointcloud(pointcloud, img, cal_info):
    """
    :param pointcloud: (N, 4) [x, y, z, intensity] or (N, 3) [x, y, z]
    :param img:
    :param cal_info:
    :return: (N, 5) [x, y, z, intensity, rgb] or (N, 4) [x, y, z, rgb]
    """
    intri_mat = cal_info["intri_mat"]
    distor = cal_info["distor"]

    # 对图像去畸变
    undistor_img = cv2.undistort(src=img, cameraMatrix=intri_mat, distCoeffs=distor)
    mask, pts_pixel = image_region_filter(pointcloud, undistor_img, cal_info)
    pointcloud = pointcloud[mask]

    # 索引颜色值
    color_np = undistor_img[np.int_(pts_pixel[:, 1]), np.int_(pts_pixel[:, 0])]

    r = np.asarray(color_np[:, 2], dtype=np.uint32)
    g = np.asarray(color_np[:, 1], dtype=np.uint32)
    b = np.asarray(color_np[:, 0], dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
    rgb_arr.dtype = np.float32
    pointcloud = np.hstack((pointcloud, rgb_arr[:, np.newaxis]))
    return pointcloud, mask


def image_region_filter(pointcloud, img, cal_info):
    # 将点云转换到像素平面上
    pts_img, z_camera = lidar_to_img_points(pointcloud[:, :3], cal_info)
    # 只保留在相机前方的点
    mask1 = z_camera > 0
    # 只保留相机FOV内的点
    pts_img = np.floor(pts_img)
    mask2 = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img.shape[1]) & \
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img.shape[0])
    mask = np.logical_and(mask1, mask2)
    pts_img = pts_img[mask].astype(np.int32)

    return mask, pts_img

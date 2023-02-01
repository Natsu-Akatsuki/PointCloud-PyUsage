import random

import open3d as o3d
from .. import visualization
import numpy as np
from colorama import Fore, Style


def plane_fitting_o3d(pointcloud, debug=False, log=True, distance_threshold=0.02, num_iterations=100):
    """
    :param distance_threshold:
    :param pointcloud:
    :param debug:
    :return:
    """

    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
    plane_model, inlier_idx = pointcloud_o3d.segment_plane(distance_threshold=distance_threshold, ransac_n=3,
                                                           num_iterations=num_iterations, probability=1.0)
    inlier_pointcloud = pointcloud[inlier_idx]

    if log:
        print(Fore.GREEN + f"[RANSAC Fitting]: {pointcloud.shape} -> {inlier_pointcloud.shape}" + Style.RESET_ALL)

    if debug:
        inlier_pointcloud_o3d = pointcloud_o3d.select_by_index(inlier_idx, invert=False)
        color_o3d = visualization.intensity_to_color_o3d(inlier_pointcloud[:, 3], is_normalized=False)
        inlier_pointcloud_o3d.colors = o3d.utility.Vector3dVector(color_o3d)

        outlier_pointcloud_o3d = pointcloud_o3d.select_by_index(inlier_idx, invert=True)
        outlier_pointcloud_o3d.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([inlier_pointcloud_o3d, outlier_pointcloud_o3d])
    return plane_model, inlier_pointcloud


def plane_fitting_ransac(pointcloud, debug=False, log=False, distance_threshold=0.02, num_iterations=100, seed=None):
    indices = np.arange(pointcloud.shape[0])
    best_plane_model = np.zeros(4, dtype=np.float32)
    max_score = 0
    random.seed(seed)

    for i in range(num_iterations):

        plane_model = np.zeros(4, dtype=np.float32)
        # 步骤一：随机采样
        sample_indices = random.sample(range(indices.shape[0]), k=3)  # 该随机采样方法比choice快10+ms
        sample_points = pointcloud[indices[sample_indices]][:, :3]

        # 步骤二：用三点拟合平面模型
        l1 = sample_points[0] - sample_points[1]
        l2 = sample_points[0] - sample_points[2]
        if np.any(l2 < 1e-6):
            continue
        quotient = l1 / l2

        # 排除三点共线的情况
        if np.isclose(quotient[0], quotient[1], atol=1e-6) and \
                np.isclose(quotient[0], quotient[2], atol=1e-6) and \
                np.isclose(quotient[1], quotient[2], atol=1e-6):
            continue

        # 法向量归一化和求取平面系数
        n = np.cross(l1, l2)
        n = n / np.linalg.norm(n)
        plane_model[:3] = n
        plane_model[3] = -sample_points[0][0] * n[0] - sample_points[0][1] * n[1] - sample_points[0][2] * n[2]

        # 步骤四：统计内点（inlier）的个数，平面的距离小于阈值的点为内点
        dis = np.fabs(pointcloud[:, :3] @ plane_model[:3] + plane_model[3])

        # 用inlier的个数作为评分
        score = indices[dis < distance_threshold].shape[0]

        # 挑选内点最多的平面作为待拟合的平面
        if score > max_score:
            max_score = score
            inlier_idx = indices[dis < distance_threshold]
            outlier_idx = indices[dis >= distance_threshold]
            best_plane_model = plane_model

    inlier_pointcloud = pointcloud[inlier_idx]

    if log:
        print(Fore.GREEN + f"[RANSAC Fitting]: {pointcloud.shape} -> {inlier_pointcloud.shape}" + Style.RESET_ALL)

    if debug:
        inlier_pointcloud_o3d = o3d.geometry.PointCloud()
        inlier_pointcloud_o3d.points = o3d.utility.Vector3dVector(inlier_pointcloud[:, :3])
        color_o3d = visualization.intensity_to_color_o3d(inlier_pointcloud[:, 3], is_normalized=False)
        inlier_pointcloud_o3d.colors = o3d.utility.Vector3dVector(color_o3d)

        outlier_pointcloud_o3d = o3d.geometry.PointCloud()
        outlier_pointcloud = pointcloud[outlier_idx]
        outlier_pointcloud_o3d.points = o3d.utility.Vector3dVector(outlier_pointcloud[:, :3])
        outlier_pointcloud_o3d.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([inlier_pointcloud_o3d])

    return best_plane_model, inlier_pointcloud

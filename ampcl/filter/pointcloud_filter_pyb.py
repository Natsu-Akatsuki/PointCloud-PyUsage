from colorama import Fore, Style

from .filter_pyb import cVoxelFilter
from .filter_pyb import cPlaneFitting
import open3d as o3d
import numpy as np
from .. import visualization


def c_voxel_filter(pointcloud, voxel_size=(0.05, 0.05, 0.05), mode="mean", log=False):
    """

    :param pointcloud:
    :param voxel_size:
    :param mode: mean （以体素中的激光点的几何中心作为采样点）or uniform（以体素中最接近体素中心的点作为采样点）
    :param log:
    :return:
    """
    if log:
        print(f"[Filtered] Before down sample the point cloud size is {pointcloud.shape[0]}")
    filtered_pointcloud = cVoxelFilter(pointcloud, voxel_size=voxel_size, mode=mode)
    if log:
        print(f"[Filtered] Before down sample the point cloud size is {filtered_pointcloud.shape[0]}")
    return filtered_pointcloud


def c_ransac_plane_fitting(pointcloud, distance_threshold=0.02,
                           use_optimize_coeff=False, random=False,
                           max_iterations=100,
                           debug=False, log=True):
    plane_model = cPlaneFitting(pointcloud, distance_threshold=distance_threshold, max_iterations=max_iterations,
                                use_optimize_coeff=use_optimize_coeff, random=random)

    if len(plane_model) == 1:
        return plane_model, pointcloud
    else:
        indices = np.arange(pointcloud.shape[0])
        dis = np.fabs(pointcloud[:, :3] @ plane_model[:3] + plane_model[3])
        inlider_idx = indices[dis < distance_threshold]
        outlier_idx = indices[dis >= distance_threshold]
        inlier_pointcloud = pointcloud[inlider_idx]

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

        return plane_model, inlier_pointcloud

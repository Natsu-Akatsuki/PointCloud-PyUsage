from ..filter import c_ransac_plane_fitting
from ..filter import passthrough_filter
from ..visualization import o3d_viewer_from_pointcloud
import open3d as o3d
import numpy as np
from .gpf_pyb import GPF


def ground_segmentation_ransac(pc_np, limit_range, distance_threshold=0.5, max_iterations=50, debug=False):
    """基于PCL的实现
    :param np.ndarray pc_np: (N, ...) [x, y, z, ...]
    :param tuple limit_range: 直通滤波的范围 qx_min, x_max, y_min, y_max, z_min, z_max)，用于选取拟合地面模型的点云
    :param float distance_threshold: 地面点的距离阈值
    :param bool debug：是否使用Open3D可视化点云
    :return:
    """
    candidate_ground_mask = passthrough_filter(pc_np, limit_range)
    candidate_ground_pointcloud = pc_np[candidate_ground_mask]
    plane_model, _ = c_ransac_plane_fitting(candidate_ground_pointcloud, distance_threshold=distance_threshold,
                                            log=False, max_iterations=max_iterations,
                                            debug=False)
    dis = np.fabs(pc_np[:, :3] @ plane_model[:3] + plane_model[3])
    ground_mask = dis < distance_threshold

    if debug:
        non_ground_mask = dis >= distance_threshold
        ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[ground_mask], show_pc=False)
        ground_pc_o3d.paint_uniform_color([1, 0, 0])
        non_ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[non_ground_mask], show_pc=False)
        non_ground_pc_o3d.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground_pc_o3d, non_ground_pc_o3d])

    return plane_model, ground_mask


def ground_segmentation_gpf(pc_np, sensor_height=1.73, inlier_threshold=0.3, iter_num=3, num_lpr=250,
                            seed_height_offset=1.2, debug=False):
    gpf = GPF(sensor_height=sensor_height, inlier_threshold=inlier_threshold, iter_num=iter_num, num_lpr=num_lpr,
              seed_height_offset=seed_height_offset)

    ground_mask = gpf.apply(pc_np[:, :3])
    if debug:
        non_ground_mask = ~ground_mask
        ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[ground_mask], show_pc=False)
        ground_pc_o3d.paint_uniform_color([1, 0, 0])
        non_ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[non_ground_mask], show_pc=False)
        non_ground_pc_o3d.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground_pc_o3d, non_ground_pc_o3d])

    return ground_mask

import numpy as np
from ..filter import c_ransac_plane_fitting
from ..filter import passthrough_filter
from ..visualization import o3d_viewer_from_pointcloud
import open3d as o3d


def ground_segmentation_ransac(pointcloud, limit_range, distance_threshold=0.5, debug=False):
    """
    :param np.ndarray pointcloud: (N, ...) [x, y, z, ...]
    :param tuple limit_range: 直通滤波的范围 qx_min, x_max, y_min, y_max, z_min, z_max)，用于选取拟合地面模型的点云
    :param float distance_threshold: 地面点的距离阈值
    :param bool debug：是否使用Open3D可视化点云
    :return:
    """
    indices = np.arange(pointcloud.shape[0])
    candidate_ground_mask = passthrough_filter(pointcloud, limit_range)
    candidate_ground_pointcloud = pointcloud[candidate_ground_mask]
    plane_model, _ = c_ransac_plane_fitting(candidate_ground_pointcloud, distance_threshold=0.5, log=False,
                                            debug=False)
    dis = np.fabs(pointcloud[:, :3] @ plane_model[:3] + plane_model[3])
    ground_mask = dis < distance_threshold
    non_ground_mask = dis >= distance_threshold

    if debug:
        ground_point_o3d = o3d_viewer_from_pointcloud(pointcloud[ground_mask], is_show=False)
        ground_point_o3d.paint_uniform_color([1, 0, 0])
        non_ground_point_o3d = o3d_viewer_from_pointcloud(pointcloud[non_ground_mask], is_show=False)
        non_ground_point_o3d.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground_point_o3d, non_ground_point_o3d])

    return plane_model, ground_mask
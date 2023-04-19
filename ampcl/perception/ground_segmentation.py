import numpy as np
import open3d as o3d

from ..filter import c_ransac_plane_fitting
from ..filter import passthrough_filter
from ..visualization import o3d_viewer_from_pointcloud


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
    non_ground_mask = dis >= distance_threshold

    if debug:
        ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[ground_mask], show_pc=False)
        ground_pc_o3d.paint_uniform_color([1, 0, 0])
        non_ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[non_ground_mask], show_pc=False)
        non_ground_pc_o3d.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground_pc_o3d, non_ground_pc_o3d])

    return plane_model, ground_mask


class GPF:
    def __init__(self, sensor_height=1.73, inlier_threshold=0.3, iter_num=3,
                 num_lpr=250, seed_height_offset=1.2):
        # 激光雷达相对于地面的高度
        self.sensor_height = sensor_height
        # 地面点的距离阈值
        self.inlier_threshold = inlier_threshold
        # 地面模型预测的迭代次数
        self.iter_num = iter_num

        # 初始种子点选取的相关阈值
        # The Lowest Point Representative (LPR) of the sorted point cloud
        self.num_lpr = num_lpr
        self.seed_height_offset = seed_height_offset

    def extract_initial_seeds(self, pc_np):
        """
        Extract initial ground seeds
        :param pc_np:
        :return:
        """

        idx = np.argsort(pc_np[:, 2])
        pc_np = pc_np[idx]
        height_mean = pc_np[:self.num_lpr, 2].mean()
        # 初始地面种子点的选取：取决于最低的种子点z值均值+高度补偿量
        ground_seed_pc_mask = pc_np[:, 2] < (height_mean + self.seed_height_offset)

        return ground_seed_pc_mask

    def noise_filter(self, pc_np):
        # remove noise caused from mirror reflection
        pc_np = pc_np[pc_np[:, 2] >= -self.sensor_height * 1.5]
        return pc_np

    def apply(self, pc_np, debug=False):
        """
        :param pc_np:
        :return:
        """
        pc_np = pc_np.copy()[:, :3]
        pc_np = self.noise_filter(pc_np)
        ground_seed_mask = self.extract_initial_seeds(pc_np)
        ground_mask = ground_seed_mask

        # coarse to fine的迭代
        for i in range(self.iter_num):
            ground_pc = pc_np[ground_mask]
            # 计算地面种子点的协方差矩阵
            coeff = np.zeros(4, dtype=np.float32)
            # (N, 3)->(3, N): (3, N)(N, 3) = (3, 3)
            cov_mat = np.cov(ground_pc.T)
            eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(cov_mat)

            # 基于观察：法向量对应的特征向量的特征值最小
            # 则假定最小特征值对应的特征向量即法向量
            n = eigenvectors[:, -1]
            n = n / np.linalg.norm(n)
            point_mean = ground_pc.mean(axis=0)

            coeff[:3] = n
            coeff[3] = -point_mean[0] * n[0] - point_mean[1] * n[1] - point_mean[2] * n[2]

            dis = np.fabs(pc_np @ coeff[:3] + coeff[3])

            ground_mask = dis < self.inlier_threshold
            non_ground_mask = dis >= self.inlier_threshold

        if debug:
            ground_pc_o3d = o3d_viewer_from_pointcloud(pc_np[ground_mask], show_pc=False)
            ground_pc_o3d.paint_uniform_color([1, 0, 0])
            non_ground_pc = o3d_viewer_from_pointcloud(pc_np[non_ground_mask], show_pc=False)
            non_ground_pc.paint_uniform_color([0, 0, 0])
            o3d.visualization.draw_geometries([ground_pc_o3d, non_ground_pc])

        return ground_mask

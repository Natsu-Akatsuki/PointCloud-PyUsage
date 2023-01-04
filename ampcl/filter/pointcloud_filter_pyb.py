from .filter_pyb import cVoxelFilter


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

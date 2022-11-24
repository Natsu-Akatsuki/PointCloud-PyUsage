from .filter_pyb import cVoxelFilter


def c_voxel_filter(pointcloud, voxel_size=(0.05, 0.05, 0.05), log=False):
    if log:
        print(f"[Filtered] Before down sample the point cloud size is {pointcloud.shape[0]}")
    filtered_pointcloud = cVoxelFilter(pointcloud, voxel_size=voxel_size)
    if log:
        print(f"[Filtered] Before down sample the point cloud size is {filtered_pointcloud.shape[0]}")
    return filtered_pointcloud

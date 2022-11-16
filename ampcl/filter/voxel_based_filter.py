import numpy as np

def voxelization(pointcloud, voxel_size):
    """
    计算每个激光点的一维索引
    :param pointcloud:
    :param voxel_size:
    :return:
    """
    limit_range = np.zeros(6)
    limit_range[0:2] = np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0])
    limit_range[2:4] = np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1])
    limit_range[4:6] = np.min(pointcloud[:, 2]), np.max(pointcloud[:, 2])

    inverse_voxel_size = 1.0 / np.array(voxel_size)

    # todo: 不能是浮点数
    x_idx = ((pointcloud[:, 0] - limit_range[0]) * inverse_voxel_size[0]).astype(np.int32)
    y_idx = ((pointcloud[:, 1] - limit_range[2]) * inverse_voxel_size[1]).astype(np.int32)
    z_idx = ((pointcloud[:, 2] - limit_range[4]) * inverse_voxel_size[2]).astype(np.int32)

    dy = (limit_range[3] - limit_range[2]) * inverse_voxel_size[1]
    dz = (limit_range[5] - limit_range[4]) * inverse_voxel_size[2]

    # 三维体素索引转换为一维体素索引
    voxel_indices1d = (x_idx * dy * dz + y_idx * dz + z_idx)
    return pointcloud, voxel_indices1d


class HashVoxel:
    def __init__(self, point):
        self.data = point
        self.point_num = 1

    def update(self, point):
        self.data += point
        self.point_num += 1


def hash_voxel_based_filter(pointcloud, voxel_size=(0.05, 0.05, 0.05), ):
    """
    :param pointcloud:
    :param voxel_size:
    :return:
    """
    print(f"[Filtered] Before down sample the point cloud size is {pointcloud.shape[0]}")

    pointcloud, voxel_indices1d = voxelization(pointcloud, voxel_size)
    feat_map = {}

    for i in range(pointcloud.shape[0]):
        # 获取激光点的体素索引，然后放进对应的哈希表中（以体素坐标为键，激光点的数据为值）
        key = voxel_indices1d[i]
        if key in feat_map:
            voxel = feat_map[key]
            voxel.update(pointcloud[i])
        else:
            voxel = HashVoxel(pointcloud[i])
            feat_map[key] = voxel

    filtered_pointcloud_size = feat_map.__len__()
    filtered_pointcloud = np.zeros((filtered_pointcloud_size, pointcloud.shape[1]), dtype=np.float32)
    for i, key in enumerate(feat_map):
        voxel = feat_map[key]
        point_num = voxel.point_num
        filtered_pointcloud[i] = voxel.data / point_num

    print(f"[Filtered] Before down sample the point cloud size is {filtered_pointcloud.shape[0]}")
    return filtered_pointcloud


def voxel_based_filter(pointcloud, voxel_size=(0.05, 0.05, 0.05)):
    print(f"[Filtered] Before down sample the point cloud size is {pointcloud.shape[0]}")

    pointcloud, voxel_indices1d = voxelization(pointcloud, voxel_size)

    # 对体素索引相同的激光点做归约操作
    arg_indices = np.argsort(voxel_indices1d)
    voxel_indices1d = voxel_indices1d[arg_indices]
    pointcloud = pointcloud[arg_indices]

    filtered_pointcloud = []
    last_voxel_index = voxel_indices1d[0]
    points_in_voxel = [pointcloud[0]]

    for i in range(1, pointcloud.shape[0]):
        cur_voxel_idx = voxel_indices1d[i]

        if cur_voxel_idx == last_voxel_index:
            points_in_voxel.append(pointcloud[i])
        else:
            point = np.mean(np.vstack(points_in_voxel), axis=0)
            filtered_pointcloud.append(point)
            points_in_voxel = [pointcloud[i]]

        last_voxel_index = cur_voxel_idx

    filtered_pointcloud = np.vstack(filtered_pointcloud)
    print(f"[Filtered] Before down sample the point cloud size is {filtered_pointcloud.shape[0]}")
    return 0

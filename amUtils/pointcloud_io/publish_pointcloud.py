import time

import numpy as np
import open3d as o3d
import struct


def load_pcd_binary_data(file_path):
    with open(file_path, 'rb') as f:
        for line in range(11):
            lines = f.readline()
            if line == 9:
                pts_num = int(lines.decode().strip('\n').split(' ')[-1])
        binary_data = f.read(pts_num * 32)

    pointcloud = np.frombuffer(binary_data, dtype=np.float32).reshape(-1, 8)
    pointcloud = pointcloud.take([0, 4, 5, 6], axis=-1)
    return pointcloud


def load_pcd_data(file_path):
    pts = []

    with open(file_path, 'r') as f:
        data = f.readline()

    pts_num = data[9].strip('\n').split(' ')[-1]
    pts_num = int(pts_num)
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        x, y, z, intensity = [eval(i) for i in xyzi[:4]]
        pts.append([x, y, z, intensity])

    assert len(pts) == pts_num
    pointcloud = np.zeros((pts_num, len(pts[0])), dtype=np.float32)
    for i in range(pts_num):
        pointcloud[i] = pts[i]
    return pointcloud


def load_pcd_data_o3d(file_path):
    """
    note: 新版本的open3D已支持读取带intensity的pcd文件，但是o3d对象不存储点云信息
    :param file_path:
    :return:
    """
    pointcloud = o3d.io.read_point_cloud(file_path)
    pointcloud = np.asarray(pointcloud.points)
    return pointcloud


def load_bin_data(file_path):
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def load_pcd_data_pcl(file_path):
    import load_pcd_file_pcl
    pointcloud = load_pcd_file_pcl.load_pcd_file(file_path, True)
    return pointcloud


if __name__ == '__main__':
    # data = load_bin_data("../data/semantic_000000.bin")[:, :3]
    # dis = np.linalg.norm(data, axis=1)
    # print(dis.max(), dis.min())
    data = load_pcd_binary_data("../data/curve.pcd")[:, :3]

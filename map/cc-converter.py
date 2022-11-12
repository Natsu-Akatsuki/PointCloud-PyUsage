import numpy as np
from ampcl.io import save_pointcloud


def load_pcd(file_path):
    """
    load pcd file exported from cloudcompare
    FIELDS intensity _ x y z _
    SIZE 4 1 4 4 4 1
    TYPE F U F F F U
    COUNT 1 12 1 1 1 4 -> （4*1+1*12+4*1+4*1+4*1+1*4) = 8*4 bytes
    """
    with open(file_path, 'rb') as f:
        for line in range(11):
            lines = f.readline()
            if line == 9:
                pts_num = int(lines.decode().strip('\n').split(' ')[-1])
        raw_data = f.read(pts_num * 32)

    pointcloud = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 8)

    # pointcloud = pointcloud.take([0, 4, 5, 6], axis=-1)
    # i, x, y, z -> x, y, z, i
    pointcloud = pointcloud[:, [4, 5, 6, 0]]
    return pointcloud


if __name__ == '__main__':
    cloud_compare_file = "pointcloud_map.cc.pcd"
    output_file = "pointcloud_map.cc.pcd"
    pc_cloudcompare = load_pcd(cloud_compare_file)
    save_pointcloud(pc_cloudcompare, output_file)

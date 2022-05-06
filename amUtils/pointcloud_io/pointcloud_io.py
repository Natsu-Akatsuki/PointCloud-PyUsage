import time

import numpy as np
import open3d as o3d
import struct


def load_cc_binary(file_path):
    """
    load pcd file exported from cloudcompare
    FIELDS intensity _ x y z _
    SIZE 4 1 4 4 4 1
    TYPE F U F F F U
    COUNT 1 12 1 1 1 4

    :usage load_ccpcd_binary_data("data/checkboard.cc.pcd")
    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as f:
        for line in range(11):
            lines = f.readline()
            if line == 9:
                pts_num = int(lines.decode().strip('\n').split(' ')[-1])
        binary_data = f.read(pts_num * 32)

    pointcloud = np.frombuffer(binary_data, dtype=np.float32).reshape(-1, 8)

    # pointcloud = pointcloud.take([0, 4, 5, 6], axis=-1)
    # i, x, y, z -> x, y, z, i
    pointcloud = pointcloud[:, [4, 5, 6, 0]]
    return pointcloud


def load_pcd(file_path):
    pts = []

    with open(file_path, 'r') as f:
        data = f.readlines()

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


def load_pcd_o3d(file_path):
    """
    note: 新版本的open3D已支持读取带intensity的pcd文件，但是o3d对象不存储点云信息
    :param file_path:
    :return:
    """
    pointcloud = o3d.io.read_point_cloud(file_path)
    pointcloud = np.asarray(pointcloud.points)
    return pointcloud


def load_bin(file_path):
    """
    :param file_path:
    :return:
    """
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def load_npy(file_path):
    """
    :param file_path:
    :return:
    """
    pointcloud = np.load(file_path)
    if not isinstance(pointcloud[0][0], np.float32):
        pointcloud = pointcloud.astype(np.float32)
        print("Attention: the pointcloud type will be cast into float32")
    if pointcloud.shape[1] > 4:
        print("Attention: the pointcloud shape is", pointcloud.shape)

    return pointcloud


def load_pcd_pcl(file_path):
    # 由于使用的是package，这里使用相对路径
    from . import load_pcd_file_pcl
    pointcloud = load_pcd_file_pcl.load_pcd_file(file_path, True)
    return pointcloud


def save_pointcloud(pointcloud_np, method="npy", file_name="pointcloud.npy"):
    """
    1) save pointcloud as npy/bin format using numpy API
    2) save pointcloud as pcd format using c++ pcl pybind
    3) save pointcloud as pcd format using open3d API
    4) save pointclodu as pcd format using python API
    :param pointcloud_np:
    :param export_format:
    :param file_name:
    :return:
    """
    if method == "npy":
        np.save(file_name, pointcloud_np)
    elif method == "pcl":
        from . import save_pcd_file_pcl
        save_pcd_file_pcl.save_pcd_file(file_name, pointcloud_np)
    elif method == "o3d":
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud_np[:, 0:3])
        o3d.io.write_point_cloud(file_name, point_cloud_o3d, write_ascii=False, compressed=True)
    elif method == "bin":
        pointcloud_np.tofile(file_name)
    elif method == "python":
        with open(file_name, 'w') as f:
            f.write("# .PCD v.7 - Point Cloud Data file format\n")
            f.write("VERSION .7\n")
            f.write("FIELDS x y z i\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH {}\n".format(pointcloud_np.shape[0]))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(pointcloud_np.shape[0]))

            # ASCII
            # f.write("DATA ascii\n")
            # for i in range(pointcloud.shape[0]):
            #     f.write(
            #         str(pointcloud[i][0]) + " " + str(pointcloud[i][1]) + " " + str(pointcloud[i][2]) + " " + str(
            #             pointcloud[i][3]) + "\n")

            f.write("DATA binary\n")
            pointcloud_np.tofile(f)


def ros_subscribe_usage(topic_name="/livox/lidar"):
    import rospy
    from sensor_msgs.msg import PointCloud2

    def pointcloud_callback(msg):
        import sensor_msgs.point_cloud2 as pc2_parser
        pointcloud_ros = pc2_parser.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pointcloud_np = np.asarray(list(pointcloud_ros), dtype=np.float32)
        # todo

    rospy.init_node('subscribe_pointcloud', anonymous=False)
    rospy.Subscriber(topic_name, PointCloud2, pointcloud_callback)
    rospy.spin()

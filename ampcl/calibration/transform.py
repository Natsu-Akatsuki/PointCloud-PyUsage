import numpy as np
from scipy.spatial.transform import Rotation

# isort: off
try:
    import rospy

    __ROS_VERSION__ = 1

except:
    try:
        import rclpy

        __ROS_VERSION__ = 2
    except:
        raise ImportError("Please install ROS2 or ROS1")

from tf2_ros import TransformException


# isort: on


def solve_frame_tf():
    """
    求坐标系之间只发生过旋转变换的变换矩阵（坐标系变换矩阵）
    :return:
    """
    lidar_frame = np.asarray([1, 0, 0,
                              0, 1, 0,
                              0, 0, 1]).reshape([3, 3])

    # camera系的基座标向量在lidar系下的表征：[x'| y'| z']
    camera_frame = np.asarray([0, 0, 1,
                               -1, 0, 0,
                               0, -1, 0]).reshape([3, 3])
    # camera_frame = TF @ lidar_frame
    rotation_mat = camera_frame @ lidar_frame.T

    # 得到欧拉角
    rotation = Rotation.from_matrix(rotation_mat)
    print(rotation.as_euler('ZYX', degrees=True))  # (-90, 0, -90) 对应于 (先Z->Y->X轴)


def ros_xyzypr_to_tf_mat(xyzypr, degrees=True, is_basis_change=True):
    """
    基于A系到B系的平移量和A系到B系的欧拉角，得到A系->B系的基变换（适用于ROS），等价于B系到A系的坐标变换
    note: 在ROS下，是先平移再旋转
    :param xyzypr:
    :param degrees:
    :param is_basis_change: 基变换 or 坐标变换
    """
    extri_matrix = np.identity(4)
    rotation = Rotation.from_euler('ZYX', xyzypr[3:], degrees=degrees)
    rot_mat = rotation.as_matrix()

    extri_matrix[:3, :3] = rot_mat
    extri_matrix[:3, 3] = np.asarray(xyzypr[:3])

    if is_basis_change:
        return extri_matrix
    else:
        return np.linalg.inv(extri_matrix)


def tf_mat_to_ros_xyzypr(tf_mat, degrees=False, is_basis_change=True):
    """
    坐标变换->ros坐标系变换（坐标系的刚体运动）
    e.g 已知激光雷达系->相机系的坐标变换，求激光雷达系->相机系的刚体变换（用xyzypr表示）
    1）先得到激光雷达系->相机系的坐标系变换
    2）再进行拆分
    :param tf_mat:
    :return:
    """
    if not is_basis_change:
        tf_mat = np.linalg.inv(tf_mat)

    xyzypr = np.zeros(6)
    xyzypr[:3] = tf_mat[:3, 3]
    xyzypr[3:] = Rotation.from_matrix(tf_mat[:3, :3]).as_euler('ZYX', degrees=degrees)
    return xyzypr


def euler_from_transformation(tf_mat):
    """
    基于变换矩阵，得到欧拉角
    :param tf_mat:
    :return:
    """

    rotation = Rotation.from_matrix(tf_mat[:3, :3])
    rotation.as_euler('ZYX', degrees=True)


def transform_box3d_frame_by_ros(self, box3d_src, target_frame, src_frame):
    """
    将src_frame的box3d转换到target_frame
    """
    box3d_target = box3d_src.copy()
    xyz = box3d_target[:, :3]
    xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

    try:
        t = self.tf_buffer.lookup_transform(
            target_frame,
            src_frame,
            rclpy.time.Time())  # 只要最新的数据
        # 获得的是基坐标变换
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        rx = t.transform.rotation.x
        ry = t.transform.rotation.y
        rz = t.transform.rotation.z
        rw = t.transform.rotation.w
        r = Rotation.from_quat([rx, ry, rz, rw])
        r = r.as_euler(seq="ZYX", degrees=False)  # 其TF变换是xyz->ypr
        extri_mat = ros_xyzypr_to_tf_mat([x, y, z, r[0], r[1], r[2]], degrees=False, is_basis_change=True)

    except TransformException as ex:
        self.get_logger().info(
            f'Could not transform {src_frame} to {target_frame}: {ex}')
        return

    xyz = np.dot(xyz, extri_mat.T)[:, :3]
    box3d_target[:, :3] = xyz[:, :3]
    return box3d_src


if __name__ == '__main__':
    # ROS的基变换
    # camera_to_lidar_xyzrpy = (0, 0, 0, 0, -90, 90)
    # lidar_to_camera_ros_xyzypr = (0, 0, 0, 90, 0, 90)
    map_to_robot_a = (3, -2.7, -6.8, -5.882, -1.453, -3.528)
    map_to_robot_b = (-10.6, 23.2, -5.7, -9.304, -1.871, -2.892)

    capture_to_robot_a = (-5.192, -4.93, 1.276, 5.37, -0.22, 0.8)
    capture_to_robot_b = (-20.89, 17.159, 2.639, 2.55, -0.55, 1.34)

    with np.printoptions(precision=2, suppress=True):
        tf_map_to_robot_a = ros_xyzypr_to_tf_mat(map_to_robot_a, is_basis_change=True)
        tf_capture_to_robot_a = ros_xyzypr_to_tf_mat(capture_to_robot_a, is_basis_change=True)
        tf_map_to_robot_b = ros_xyzypr_to_tf_mat(map_to_robot_b, is_basis_change=True)
        tf_map_to_capture = tf_map_to_robot_a @ np.linalg.inv(tf_capture_to_robot_a)
        tf_capture_to_map = np.linalg.inv(tf_map_to_capture)

    capture_to_robot_b_estimate = tf_capture_to_map @ tf_map_to_robot_b
    estimate = tf_mat_to_ros_xyzypr(capture_to_robot_b_estimate, degrees=True, is_basis_change=True)
    pass
    # 求激光雷达系到相机的坐标变换->激光雷达系到相机系的刚体变换在ROS下的表示
    # tf_mat = np.array([0, -1, 0, 0,
    #                    0, 0, -1, 0,
    #                    1, 0, 0, 0,
    #                    0, 0, 0, 1]).reshape([4, 4])

    # with np.printoptions(precision=2, suppress=True):
    #     ros_xyzypr = tf_mat_to_ros_xyzypr(mat1, degrees=True, is_basis_change=True)
    #     print("【ROS基变换 xyzypr】激光雷达系到相机系: \n", ros_xyzypr)

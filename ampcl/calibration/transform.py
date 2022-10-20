import numpy as np
from scipy.spatial.transform import Rotation


def frame_tf_to_basic_change():
    """
    坐标系变换->基变换
    :return:
    """
    mat44 = np.array(
        [0, 0, -1, 0,
         -1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 0, 1]).reshape(4, 4)
    translation = np.array([0, 0, 0])
    mat44[0:3, 3] = translation

    print("origin: \n", mat44)
    print("cvt: \n", np.linalg.inv(mat44))


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


def ros_frame_tf_to_basic_change(tx_ty_tz_ypr=(0, 0, 0, 0, 0, 0)):
    """
    ros坐标系变换->基变换
    :return:
    """

    rotation = Rotation.from_euler('ZYX', tx_ty_tz_ypr[3:], degrees=True)
    rot_mat = rotation.as_matrix()

    extri_matrix = np.identity(4)
    extri_matrix[:3, :3] = rot_mat
    extri_matrix[:3, 3] = tx_ty_tz_ypr[:3]
    return np.linalg.inv(extri_matrix)


if __name__ == '__main__':
    tx_ty_tz_ypr = (0, 0, 0, -90, 0, -90)
    with np.printoptions(precision=2, suppress=True):
        print(ros_frame_tf_to_basic_change(tx_ty_tz_ypr))

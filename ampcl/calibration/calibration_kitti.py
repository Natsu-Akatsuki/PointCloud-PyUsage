import numpy as np


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class KITTICalibration():
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            cal = get_calib_from_file(calib_file)
            self.P2 = cal['P2']  # 3 x 4
            self.fu = self.P2[0, 0]
            self.fv = self.P2[1, 1]

            # 3x4->4x4
            lidar_to_ref_mat = cal['Tr_velo2cam']
            lidar_to_ref_mat = np.vstack((lidar_to_ref_mat, np.zeros((1, 4), dtype=np.float32)))  # 4 x 4
            lidar_to_ref_mat[3, 3] = 1

            # 3x3->4x4
            ref_to_cam2_mat = np.zeros((4, 4), dtype=np.float32)
            ref_to_cam2_mat[:3, :3] = cal['R0']
            ref_to_cam2_mat[3, 3] = 1

            # 参考系相机->2号相机的平移向量
            self.tx = self.P2[0, 3] / self.fu
            self.ty = self.P2[1, 3] / self.fv
            ref_to_cam2_mat[:3, 3] = np.array([self.tx, self.ty, 0], dtype=np.float32)

            # 激光雷达系->0号相机系->2号相机
            self.extri_mat = ref_to_cam2_mat @ lidar_to_ref_mat

            # 2号相机内参
            self.intri_mat = self.P2[:, :3]
            # note: KITTI三维目标检测数据集已去畸变
            self.distor = np.zeros((5, 1), dtype=np.float32)

            cal_info = {'extri_mat': self.extri_mat,
                        'intri_mat': self.intri_mat,
                        'distor': self.distor}

        else:
            cal_info = calib_file

        self.cal_info = cal_info

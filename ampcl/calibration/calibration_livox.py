import numpy as np


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    intri_mat = np.vstack((np.asarray(lines[1].strip().split(), dtype=np.float32),
                           np.asarray(lines[2].strip().split(), dtype=np.float32),
                           np.asarray(lines[3].strip().split(), dtype=np.float32),
                           ))
    distor = np.asarray(lines[6].strip().split(), dtype=np.float32)

    extri_mat = np.vstack((np.asarray(lines[9].strip().split(), dtype=np.float32),
                           np.asarray(lines[10].strip().split(), dtype=np.float32),
                           np.asarray(lines[11].strip().split(), dtype=np.float32),
                           np.asarray(lines[12].strip().split(), dtype=np.float32),
                           ))

    return {'intri_matrix': intri_mat,  # (3,3)
            'distor': distor,  # 4
            'extri_matrix': extri_mat  # (4,4)
            }


class LivoxCalibration():
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.intri_mat = calib['intri_mat']
        self.distor = calib['distor']
        self.extri_mat = calib['extri_mat']

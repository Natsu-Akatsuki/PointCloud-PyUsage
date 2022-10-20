import numpy as np

from .calibration import Calibration


class LivoxCalibration(Calibration):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.intri_matrix = calib['intri_matrix']
        self.distor = calib['distor']
        self.extri_matrix = calib['extri_matrix']

    @staticmethod
    def get_calib_from_file(calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        intri_matrix = np.vstack((np.asarray(lines[1].strip().split(), dtype=np.float32),
                                  np.asarray(lines[2].strip().split(), dtype=np.float32),
                                  np.asarray(lines[3].strip().split(), dtype=np.float32),
                                  ))
        distor = np.asarray(lines[6].strip().split(), dtype=np.float32)

        extri_matrix = np.vstack((np.asarray(lines[9].strip().split(), dtype=np.float32),
                                  np.asarray(lines[10].strip().split(), dtype=np.float32),
                                  np.asarray(lines[11].strip().split(), dtype=np.float32),
                                  np.asarray(lines[12].strip().split(), dtype=np.float32),
                                  ))

        return {'intri_matrix': intri_matrix,  # (3,3)
                'distor': distor,  # 4
                'extri_matrix': extri_matrix  # (4,4)
                }

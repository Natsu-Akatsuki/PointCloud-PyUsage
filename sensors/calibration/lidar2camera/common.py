import numpy as np


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    intri_matrix = np.vstack((np.asarray(lines[1].strip().split(','), dtype=np.float32),
                              np.asarray(lines[2].strip().split(','), dtype=np.float32),
                              np.asarray(lines[3].strip().split(','), dtype=np.float32),
                              ))
    distor = np.asarray(lines[7].strip().split(','), dtype=np.float32)

    xyz_ypr = np.asarray(lines[11].strip().split(','), dtype=np.float32).tolist()

    return {'intri_matrix': intri_matrix,  # (3,3)
            'distor': distor,  # (1, 5)
            'xyz_ypr': xyz_ypr  # (1, 7)
            }

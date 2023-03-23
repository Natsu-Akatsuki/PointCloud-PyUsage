# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os
import yaml
from easydict import EasyDict as edict


def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show


def get_threshold(dataset, det_name):
    # used for visualization only as we want to remove some false positives, also can be
    # used for KITTI 2D MOT evaluation which uses a single operating point
    # obtained by observing the threshold achieving the highest MOTA on the validation set

    if dataset == 'KITTI':
        if det_name == 'pointrcnn':
            return {'Car': 3.240738, 'Pedestrian': 2.683133, 'Cyclist': 3.645319}
        else:
            assert False, 'error, detection method not supported for getting threshold' % det_name
    elif dataset == 'nuScenes':
        if det_name == 'megvii':
            return {'Car': 0.262545, 'Pedestrian': 0.217600, 'Truck': 0.294967, 'Trailer': 0.292775,
                    'Bus': 0.440060, 'Motorcycle': 0.314693, 'Bicycle': 0.284720}
        if det_name == 'centerpoint':
            return {'Car': 0.269231, 'Pedestrian': 0.410000, 'Truck': 0.300000, 'Trailer': 0.372632,
                    'Bus': 0.430000, 'Motorcycle': 0.368667, 'Bicycle': 0.394146}
        else:
            assert False, 'error, detection method not supported for getting threshold' % det_name
    else:
        assert False, 'error, dataset %s not supported for getting threshold' % dataset

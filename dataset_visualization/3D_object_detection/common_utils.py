import colorsys
import pickle

import yaml
from easydict import EasyDict


# isort: on

class Config:
    def __init__(self, kitti_cfg_file):
        cfg_dict = {}
        with open(kitti_cfg_file, 'r') as f:
            kitti_cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg_dict.update(kitti_cfg)
        cfg_dict = EasyDict(cfg_dict)

        # DatasetParam
        self.dataset_dir = cfg_dict.DatasetParam.dataset_dir
        self.image_dir_path = cfg_dict.DatasetParam.image_dir_path
        self.pointcloud_dir_path = cfg_dict.DatasetParam.pointcloud_dir_path
        self.label_dir_path = cfg_dict.DatasetParam.label_dir_path
        self.prediction_dir_path = cfg_dict.DatasetParam.prediction_dir_path
        self.calib_dir_path = cfg_dict.DatasetParam.calib_dir_path

        # ROSParam
        self.vanilla_pointcloud_topic = cfg_dict.ROSParam.vanilla_pointcloud_topic
        self.color_pointcloud_topic = cfg_dict.ROSParam.color_pointcloud_topic
        self.img_plus_box2d_topic = cfg_dict.ROSParam.img_plus_box2d_topic
        self.img_plus_box3d_topic = cfg_dict.ROSParam.img_plus_box3d_topic
        self.box3d_topic = cfg_dict.ROSParam.box3d_topic
        self.frame_id = cfg_dict.ROSParam.frame_id

        # VisParam
        self.auto_update = cfg_dict.VisParam.auto
        self.update_time = cfg_dict.VisParam.update_time


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def id_to_color(cls_id):
    color_maps = {-1: [0.5, 0.5, 0.5],
                  0: [1.0, 0.0, 0.0],
                  1: [0.0, 1.0, 0.0],
                  2: [0.0, 0.0, 1.0],
                  3: [1.0, 0.8, 0.0],
                  4: [1.0, 0.0, 1.0]
                  }
    if cls_id not in color_maps.keys():
        return [0.5, 1.0, 0.5]

    return color_maps[cls_id]


def open_pkl_file(infos_path):
    with open(str(infos_path), 'rb') as f:
        infos = pickle.load(f)
    return infos


def generate_colors():
    """
    Aadapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    :return list of colors (each color is a list of len=3)
    """
    N = 30
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6,
            10]
    colors = [list(colors[idx]) for idx in perm]
    return colors

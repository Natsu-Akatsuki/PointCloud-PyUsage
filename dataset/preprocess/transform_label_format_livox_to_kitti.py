import time

import numpy as np
from ampcl.calibration import calibration_template
from ampcl.calibration.calibration_livox import LivoxCalibration
from ampcl import filter
from ampcl import io
from pathlib import Path

from tqdm import tqdm


def get_2D_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object2d(line) for line in lines]
    return objects


def get_3D_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def class_name_mapping(class_name, object_state=None):
    """ 类别合并
    :param class_name:
    :return:
    """
    type_to_id = {'car': 'Car',
                  'police_car': 'Car',
                  }
    if class_name == 'bicycle':
        if object_state == 'Have_a_rider':
            return 'Cyclist'
        else:
            return 'Cyclist'
    if class_name == 'motorcycle':
        if object_state == 'Have_a_rider':
            return 'Cyclist'
        else:
            return 'Cyclist'
    if class_name == 'pedestrians':
        if object_state == 'Sit_or_lie_down':
            return 'person_sitting'
        else:
            return 'Pedestrian'

    if class_name not in type_to_id.keys():
        return class_name
    return type_to_id[class_name]


class Object2d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.track_id = int(label[0]) if label[0] != 'None' else -1
        self.cls_type = class_name_mapping(label[1])
        self.cls_id = cls_type_to_id(self.cls_type)
        self.vis_percentage = label[2]
        self.object_state = label[3]
        # 左上和右下角点坐标
        self.box2d = np.asarray(label[4:8], dtype=np.float32)


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.track_id = int(label[0])
        self.vis_percentage = label[2]
        self.object_state = label[3]
        self.cls_type = class_name_mapping(label[1], self.object_state)
        self.cls_id = cls_type_to_id(self.cls_type)
        self.corner3d_lidar = np.asarray(label[4:], dtype=np.float32).reshape(-1, 3)
        self.l, self.w, self.h, self.loc_lidar, self.yaw_lidar = self.livox_to_box3d_lidar_format(self.corner3d_lidar)
        self.box3d_lidar = np.hstack((self.loc_lidar[:3], self.l, self.w, self.h, self.yaw_lidar))

    @staticmethod
    def livox_to_box3d_lidar_format(corners_lidar):
        """标签格式转换
        相关角度计算与下计算有出入，源于航向角的定义不同
        https://github.com/Livox-SDK/livox-dataset-devkit/blob/master/src/pcd_single_read/src/pcd_single_read_node.cpp
        :param corners_lidar:
        :return:
        """
        l = np.linalg.norm((corners_lidar[0] - corners_lidar[1]))
        w = np.linalg.norm((corners_lidar[0] - corners_lidar[3]))
        h = np.linalg.norm((corners_lidar[0] - corners_lidar[4]))
        cx = np.average(corners_lidar[:, 0])
        cy = np.average(corners_lidar[:, 1])
        cz = np.average(corners_lidar[:, 2])
        loc = np.array((cx, cy, cz), dtype=np.float32)
        yaw = np.arctan2(corners_lidar[1][1] - corners_lidar[0][1],
                         corners_lidar[1][0] - corners_lidar[0][0])
        return l, w, h, loc, yaw


class Transformer():
    def __init__(self):
        dataset_dir = "/home/helios/mnt/dataset/livox_dataset/training/"
        dataset_dir = Path(dataset_dir)

        calib_file = str(Path(dataset_dir) / "calib/config.txt")
        lidar_label_dir = Path(dataset_dir) / "label/lidar"
        image_label_dir = Path(dataset_dir) / "label/image"
        output_label_dir = Path(dataset_dir) / "label_2"
        output_label_dir.mkdir(exist_ok=True)
        pointcloud_dir = Path(dataset_dir) / "horizon"
        label_length = len(list(image_label_dir.iterdir()))

        num_filtered_based_on_img_width = 0
        num_filtered_based_on_img_height = 0
        num_filtered_based_on_point = 0
        num_filtered_based_on_range = 0
        num_filtered_based_on_lwh = 0
        num_supplement_label = 0
        num_append_dont_care = 0

        for i in tqdm(range(label_length)):
            objects_3d = get_3D_objects_from_label(str(lidar_label_dir / f"{i:06d}.txt"))
            objects_2d = get_2D_objects_from_label(str(image_label_dir / f"{i:06d}.txt"))
            pointcloud = io.load_pointcloud(str(pointcloud_dir / f"{i:06d}.bin"))
            self.livox_calibration = LivoxCalibration(calib_file)

            cal_info = {}
            cal_info["intri_mat"] = self.livox_calibration.intri_matrix
            cal_info["extri_mat"] = self.livox_calibration.extri_matrix
            cal_info["distor"] = self.livox_calibration.distor
            self.cal_info = cal_info

            kitti_annos = []

            object3d_without_2d_info = []
            object2d_without_3d_info = []
            for object2d in objects_2d:
                object2d_without_3d_info.append(object2d.track_id)

            for object3d in objects_3d:
                # 移除长宽高有问题的标签
                if object3d.cls_type == "Pedestrian" and (object3d.l > 2.0 or object3d.w > 2.0 or object3d.h > 2.5):
                    num_filtered_based_on_lwh += 1
                    continue
                if object3d.cls_type == "Cyclist" and (
                        object3d.l < 0.5 or object3d.l > 2.5 or object3d.w > 2.5 or object3d.h > 2.5):
                    num_filtered_based_on_lwh += 1
                    continue
                # 移除激光点较少的三维框
                if self.points_in_box(pointcloud, object3d.box3d_lidar) < 5:
                    num_filtered_based_on_lwh += 1
                    num_filtered_based_on_point += 1
                    continue
                # 移除非检测范围的三维框
                if object3d.box3d_lidar[0] > 70.4 or object3d.box3d_lidar[0] < 0 or \
                        object3d.box3d_lidar[1] > 40 or object3d.box3d_lidar[2] < -40:
                    num_filtered_based_on_range += 1
                    continue

                object3d_without_2d_info.append(object3d.track_id)
                for object2d in objects_2d:
                    if object2d.track_id == object3d.track_id:
                        # 移除在对应图像上太小的框
                        if (object2d.box2d[3] - object2d.box2d[1]) < 25:
                            num_filtered_based_on_img_width += 1
                            continue
                        # 移除二维框有问题的标签
                        if (object2d.box2d[2] - object2d.box2d[0]) <= 1:
                            num_filtered_based_on_img_height += 1
                            continue

                        kitti_anno = self.merge_objects(object2d, object3d)
                        kitti_annos.append(kitti_anno)
                        object3d_without_2d_info.remove(object3d.track_id)
                        object2d_without_3d_info.remove(object2d.track_id)

            # 只有三维信息的：
            for object3d in objects_3d:
                if object3d.track_id in object3d_without_2d_info:
                    num_supplement_label += 1

                    proj_corners2d, proj_corners3d = calibration_template.corners_3d_to_2d(object3d.corner3d_lidar,
                                                                                           self.cal_info,
                                                                                           frame="lidar")
                    object3d.box2d = proj_corners2d

                    if (object3d.box2d[3] - object3d.box2d[1]) < 25:
                        num_filtered_based_on_img_width += 1
                        continue
                    if (object3d.box2d[2] - object3d.box2d[0]) <= 1:
                        num_filtered_based_on_img_height += 1
                        continue

                    kitti_anno = self.use_only_3d_object3d(object3d)
                    kitti_annos.append(kitti_anno)

            for object2d in objects_2d:
                if object2d.cls_id in object2d_without_3d_info:
                    kitti_anno = self.create_dont_care_region_label(object2d)
                    num_append_dont_care += 1
                    kitti_annos.append(kitti_anno)

            np.savetxt(str(output_label_dir / f"{i:06d}.txt"), kitti_annos, fmt="%s")

        print(f"num_filtered_based_on_width: {num_filtered_based_on_img_height}")
        print(f"num_filtered_based_on_height: {num_filtered_based_on_img_width}")
        print(f"num_filtered_based_on_point: {num_filtered_based_on_point}")
        print(f"num_filtered_based_on_range: {num_filtered_based_on_range}")
        print(f"num_filtered_based_on_lwh: {num_filtered_based_on_lwh}")
        print(f"num_append_dont_care: {num_append_dont_care}")
        print(f"共补充了{num_supplement_label}个缺失标签")

    def use_only_3d_object3d(self, object3d):
        cls_type = object3d.cls_type
        truncation = 0
        occlusion = 3  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown

        box2d = object3d.box2d

        vis_percentage = object3d.vis_percentage
        x, y, z, l, w, h, yaw = self.livox_calibration.lidar_to_camera_lwh3d(
            object3d.box3d_lidar.reshape(1, -1)).squeeze()
        alpha = -np.arctan2(-object3d.box3d_lidar[1], object3d.box3d_lidar[0]) + yaw
        ry = float(yaw)
        level = self.get_kitti_obj_level(box2d, vis_percentage)

        kitti_anno = [cls_type, truncation, occlusion, alpha,
                      box2d[0], box2d[1], box2d[2], box2d[3],
                      h, w, l, x, y, z, ry, level
                      ]

        return kitti_anno

    def create_dont_care_region_label(self, object2d):
        box2d = object2d.box2d
        kitti_anno = ["DontCare", -1, -1, -10,
                      box2d[0], box2d[1], box2d[2], box2d[3],
                      -1, -1, -1, -1000, -1000, -1000, -10, -1
                      ]

        return kitti_anno

    def merge_objects(self, object2d, object3d):
        cls_type = object2d.cls_type
        truncation = 0
        occlusion = 3  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        box2d = object2d.box2d
        vis_percentage = object2d.vis_percentage
        x, y, z, l, w, h, yaw = self.livox_calibration.lidar_to_camera_lwh3d(
            object3d.box3d_lidar.reshape(1, -1)).squeeze()
        alpha = -np.arctan2(-object3d.box3d_lidar[1], object3d.box3d_lidar[0]) + yaw
        ry = float(yaw)
        level = self.get_kitti_obj_level(box2d, vis_percentage)
        kitti_anno = [cls_type, truncation, occlusion, alpha,
                      box2d[0], box2d[1], box2d[2], box2d[3],
                      h, w, l, x, y, z, ry, level
                      ]

        return kitti_anno

    def points_in_box(self, pointcloud, box, margin=0.1):
        point_indices = filter.get_indices_of_points_inside(pointcloud, box, margin=margin)
        return point_indices.shape[0]

    def get_kitti_obj_level(self, box2d, vis_percentage):
        """
        高度大于60+可见区域为60-100
        高度大于40+可见区域为40-100
        高度大于40+可见区域为0—40
        :param box2d:
        :param vis_percentage:
        :return:
        """
        height = float(box2d[3]) - float(box2d[1]) + 1

        if height >= 40 and vis_percentage == "81—100" or vis_percentage == "61—80":
            level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and vis_percentage == "41—60" \
                or vis_percentage == "61—80" or vis_percentage == "81—100":
            level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and vis_percentage == "0—40":
            level_str = 'Hard'
            return 2  # Hard
        else:
            level_str = 'UnKnown'
            return -1


if __name__ == '__main__':
    transformer = Transformer()

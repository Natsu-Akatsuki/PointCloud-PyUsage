from pathlib import Path

import numpy as np

from . import calibration
from . import object3d


def get_objects_from_label(label_file, cal_info):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [KITTIObject3d(line, cal_info) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class KITTIObject3d(object):
    def __init__(self, line, cal_info):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()
        self.corner3d_cam = self.box3d_cam_to_corner()
        self.corner3d_lidar = calibration.camera_to_lidar_points(self.corner3d_cam, cal_info)
        self.corner3d_lidar_pixel, _, _ = calibration.cam_to_pixel(self.corner3d_cam, cal_info)

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def box3d_cam_to_corner(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str


def kitti_object_to_pcdet_object(label_path, cal_info):
    if not Path(label_path).exists():
        return None
    obj_list = get_objects_from_label(label_path, cal_info)
    if len(obj_list) == 0:
        return None

    # use template variable to improve readability
    annotations = {'name': np.array([obj.cls_type for obj in obj_list]),
                   'bbox': np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                   'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                   'location': np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0),
                   'rotation_y': np.array([obj.ry for obj in obj_list]),
                   'score': np.array([obj.score for obj in obj_list]),
                   'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                   'box2d': np.array([obj.box2d for obj in obj_list], np.float32),
                   'corner3d_cam': np.array([obj.corner3d_cam for obj in obj_list], np.float32),
                   'corner3d_lidar': np.array([obj.corner3d_lidar for obj in obj_list], np.float32),
                   'corner3d_lidar_pixel': np.array([obj.corner3d_lidar_pixel for obj in obj_list], np.float32)
                   }

    # 获取非DT的标签个数
    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(annotations['name'])
    # 有效标签的索引
    # note: Don't care都是在最后的，所以可以用这种方式处理
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)

    loc_cam = annotations['location'][:num_objects]
    dims = annotations['dimensions'][:num_objects]
    rots = annotations['rotation_y'][:num_objects]
    score = annotations['score'][:num_objects]

    # points from camera frame->lidar frame
    loc_lidar = calibration.camera_to_lidar_points(loc_cam, cal_info)
    class_name = annotations['name'][:num_objects]
    box2d = annotations['box2d'][:num_objects]
    corner3d_cam = annotations['corner3d_cam'][:num_objects]
    corner3d_lidar = annotations['corner3d_lidar'][:num_objects]
    corner3d_lidar_pixel = annotations['corner3d_lidar_pixel'][:num_objects]

    class_id = []
    for i in range(class_name.shape[0]):
        class_id.append(int(cls_type_to_id(class_name[i])))
    class_id = np.asarray(class_id, dtype=np.int32).reshape(-1, 1)

    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2  # bottom center -> geometry center

    box3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]), class_id], axis=1)
    box3d_camera = np.concatenate([h, w, l, loc_cam, rots[..., np.newaxis], class_id], axis=1)

    infos = {
        'box2d': box2d,
        'box3d_lidar': box3d_lidar,
        'box3d_camera': box3d_camera,
        'corner3d_cam': corner3d_cam,
        'corner3d_lidar': corner3d_lidar,
        'corner3d_lidar_pixel': corner3d_lidar_pixel,
        'score': score,
    }
    return infos


def save_kitti_prediction(pred_dict, cal_info, img_shape, output_path=None):
    """导出KITTI格式的预测结果
    :param cal_info:
    :param pred_dict:
    :param output_path:
    :return:
    """
    pred_boxes = pred_dict['pred_boxes']
    pred_scores = pred_dict['pred_scores']

    box3d_cam = object3d.box3d_lidar_to_cam(pred_boxes, cal_info)
    box2d4c = object3d.box3d_cam_to_2d4c(
        box3d_cam, cal_info, img_shape=img_shape
    )

    class_names = np.array(['Car', 'Pedestrian', 'Cyclist'])
    pred_dict['name'] = np.array(class_names)[np.int32(pred_boxes[:, 7] - 1)]
    pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + box3d_cam[:, 6]
    pred_dict['bbox'] = box2d4c
    pred_dict['dimensions'] = box3d_cam[:, 3:6]
    pred_dict['location'] = box3d_cam[:, 0:3]
    pred_dict['rotation_y'] = box3d_cam[:, 6]
    pred_dict['score'] = pred_scores
    pred_dict['boxes_lidar'] = pred_boxes

    if output_path is not None:
        cur_det_file = output_path
        with open(cur_det_file, 'w') as f:
            bbox = pred_dict['bbox']
            loc = pred_dict['location']
            dims = pred_dict['dimensions']  # lhw -> hwl

            for idx in range(len(bbox)):
                print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                      % (pred_dict['name'][idx], pred_dict['alpha'][idx],
                         bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                         dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                         loc[idx][1], loc[idx][2], pred_dict['rotation_y'][idx],
                         pred_dict['score'][idx]), file=f)

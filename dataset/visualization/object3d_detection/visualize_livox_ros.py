# isort: off

try:
    import rospy

    __ROS_VERSION__ = 1


    class Node:
        def __init__(self, node_name):
            rospy.init_node(node_name)
except:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile

        __ROS_VERSION__ = 2
    except:
        raise ImportError("Please install ROS2 or ROS1")

import time
from pathlib import Path

import cv2
import numpy as np
import std_msgs.msg
from ampcl.calibration import calibration
from ampcl.calibration import object3d
from ampcl.calibration.calibration_livox import LivoxCalibration
from ampcl.calibration.object3d_kitti import get_objects_from_label
from ampcl.io import load_pointcloud
from ampcl.perception.ground_segmentation import ground_segmentation_ransac
from ampcl.ros import marker, np_to_pointcloud2, publisher
from ampcl.tracking.ab3dmot import AB3DMOT
from ampcl.visualization import color_o3d_to_color_ros, paint_box2d_on_img, paint_box3d_on_img
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from tqdm import tqdm
from visualization_msgs.msg import Marker, MarkerArray

import common_utils
from ampcl.detection import SimpleYolo


# isort: on


class Visualization(Node):
    def __init__(self):
        super().__init__("visualization")
        cfg = common_utils.Config("config/livox.yaml")

        # Data
        dataset_dir = Path(cfg.dataset_dir)
        self.image_dir_path = dataset_dir / Path(cfg.image_dir_path)
        self.pointcloud_dir_path = dataset_dir / cfg.pointcloud_dir_path
        self.label_dir_path = dataset_dir / cfg.label_dir_path
        self.prediction_dir_path = dataset_dir / cfg.prediction_dir_path
        self.calib_dir_path = dataset_dir / cfg.calib_dir_path
        self.frame_id = cfg.frame_id
        self.limit_range = [0.0, 99.6, -44.8, 44.8, -2.0, 2.0]
        self.do_track = False
        self.do_fusion = False

        if self.do_fusion:
            self.yolo_model = SimpleYolo()

        # ROS
        if __ROS_VERSION__ == 1:
            self.pointcloud_pub = rospy.Publisher(cfg.pointcloud_topic, PointCloud2, queue_size=1, latch=True)
            self.img_plus_box2d_pub = rospy.Publisher(cfg.image_topic, Image, queue_size=1, latch=True)
            self.box3d_pub = rospy.Publisher(cfg.bounding_box_topic, MarkerArray, queue_size=1, latch=True)
        if __ROS_VERSION__ == 2:
            latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

            self.pc_in_pub = self.create_publisher(PointCloud2, "/debug/pointcloud/in_region", latching_qos)
            self.pc_out_pub = self.create_publisher(PointCloud2, "/debug/pointcloud/out_region", latching_qos)

            self.color_pointcloud_pub = self.create_publisher(PointCloud2, cfg.color_pointcloud_topic, latching_qos)
            self.img_plus_box2d_pub = self.create_publisher(Image, cfg.img_plus_box2d_topic, latching_qos)
            self.img_plus_box3d_pub = self.create_publisher(Image, cfg.img_plus_box3d_topic, latching_qos)
            self.box3d_pub = self.create_publisher(MarkerArray, cfg.box3d_topic, latching_qos)

            distance_marker = marker.create_distance_marker(frame_id=self.frame_id, distance_delta=10)
            distance_marker_pub = self.create_publisher(MarkerArray, "/debug/distance_marker", latching_qos)
            distance_marker_pub.publish(distance_marker)

        self.bridge = CvBridge()

        # Visualization
        self.auto_update = cfg.auto_update
        self.update_time = cfg.update_time

        # Calibration
        calib_path = "{}/{}.txt".format(self.calib_dir_path, "config")
        calib = LivoxCalibration(calib_path)
        cal_info = {}
        cal_info["intri_mat"] = calib.intri_matrix
        cal_info["extri_mat"] = calib.extri_matrix
        cal_info["distor"] = calib.distor
        self.cal_info = cal_info

        # Tracker
        self.car_tracker = AB3DMOT(category="Car")
        self.pedestrian_tracker = AB3DMOT(category="Pedestrian")
        self.cyclist_tracker = AB3DMOT(category="Cyclist")

        self.iter_dataset_with_kitti_label()

    def pub_color_pointcloud_ros(self, header, pointcloud, img):
        """
        给点云上色
        :param header:
        :param pointcloud:
        :param img:
        """
        if self.color_pointcloud_pub.get_subscription_count() > 0:
            color_pointcloud_in_image, mask = calibration.paint_pointcloud(pointcloud, img,
                                                                           self.cal_info)
            # 将不能投影到图像上的点云的颜色设置为黑色
            pointcloud_not_in_image = pointcloud[~mask]
            rgb_arr = np.zeros((pointcloud_not_in_image.shape[0]), dtype=np.float32)
            pointcloud_not_in_image = np.hstack((pointcloud_not_in_image, rgb_arr[:, np.newaxis]))
            color_pointcloud = np.vstack((color_pointcloud_in_image, pointcloud_not_in_image))
            color_pointcloud_msg = np_to_pointcloud2(color_pointcloud, header, field="xyzirgb")
            self.color_pointcloud_pub.publish(color_pointcloud_msg)

    def track(self, detections):
        trackers = np.zeros((0, 9))
        object_classes = [1, 2, 3]
        object_trackers = [self.car_tracker, self.pedestrian_tracker, self.cyclist_tracker]
        for i, obj_class in enumerate(object_classes):
            obj_trackers, _ = object_trackers[i].track(detections[detections[:, 7] == obj_class][:, :7])
            if len(obj_trackers) != 0:
                obj_trackers = np.hstack(
                    (np.array(obj_trackers).reshape(-1, 8),
                     np.full(len(obj_trackers), fill_value=obj_class).reshape(-1, 1)))
                obj_trackers[:, 7], obj_trackers[:, 8] = obj_trackers[:, 8].copy(), obj_trackers[:, 7].copy()
                trackers = np.vstack((trackers, obj_trackers))
        return trackers

    def predict_and_publish(self, pointcloud, img, gt_infos, pred_infos=None):

        if __ROS_VERSION__ == 1:
            stamp = rospy.Time.now()
        if __ROS_VERSION__ == 2:
            stamp = self.get_clock().now().to_msg()

        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = self.frame_id

        # 图像+2D框
        gt_box2d_img, gt_box3d_lidar = gt_infos['box2d'], gt_infos['box3d_lidar']
        # img_plus_box2d = paint_box2d_on_img(img.copy(), gt_box2d_img, cls_idx=gt_box3d_lidar[:, 7])

        # 二维目标检测
        if 0:
            box2d_infer = self.yolo_model.infer(img, keep_idx=[0, 1, 2, 3, 5, 7])
            id_remap = {
                0: 2,  # person -> Pedestrian
                1: 3,  # bicycle -> Cyclist
                2: 1,  # car->Car
                3: 3,  # motorcycle->Cyclist
                5: 1,  # bus -> Car
                7: 1  # truck -> Car
            }
            for i in range(box2d_infer.shape[0]):
                box2d_infer[i, 5] = id_remap[box2d_infer[i, 5]]
            img_plus_box2d = paint_box2d_on_img(img.copy(), box2d_infer, cls_idx=box2d_infer[:, 5])
        else:
            img_plus_box2d = paint_box2d_on_img(img.copy(), gt_box2d_img, cls_idx=gt_box3d_lidar[:, 7])
        publisher.publish_img(img_plus_box2d, self.img_plus_box2d_pub, self.bridge, stamp, self.frame_id)

        # 彩色点云
        self.pub_color_pointcloud_ros(header, pointcloud, img)

        # 背景点，e.g. 地面点
        colors_np = np.full((pointcloud.shape[0], 3), np.array([0.8, 0.8, 0.8]))
        limit_range = (0, 50, -10, 10, -2.5, -1.0)
        _, ground_idx = ground_segmentation_ransac(pointcloud, limit_range, distance_threshold=0.5, debug=False)
        colors_np[ground_idx] = [0.61, 0.46, 0.33]

        # 创建marker
        box3d_marker_array = MarkerArray()
        empty_marker = Marker()
        empty_marker.action = Marker.DELETEALL
        box3d_marker_array.markers.append(empty_marker)

        marker.create_gt_detection_box3d_marker_array(pointcloud, gt_box3d_lidar.copy(),
                                                      pc_color=colors_np,
                                                      stamp=stamp,
                                                      frame_id=self.frame_id,
                                                      box3d_marker_array=box3d_marker_array,
                                                      box3d_ns="gt/detection/box3d"
                                                      )
        if pred_infos is not None:
            pred_box3d_lidar = pred_infos['box3d_lidar']
            score = pred_infos['score']
            mask = None
        if 0:
            # box3d marker: prediction for detection
            box2d_img = gt_infos['box2d']
            cls_id = gt_infos['box3d_lidar'][:, 7].reshape(-1, 1)
            box2d_img = np.hstack((box2d_img, cls_id))

            box2d_lidar = pred_infos['proj_corners2d_list']
            cls_id = pred_infos['box3d_lidar'][:, 7].reshape(-1, 1)
            box2d_lidar = np.hstack((box2d_lidar, cls_id))

            mask = self.filter_detetion_by_img_decision(box2d_img, box2d_lidar, iou_threshold=0.4)
            pred_box3d_lidar = pred_infos['box3d_lidar'][mask]
            score = pred_infos['score'][mask]

        # 图像+投影三维框
        if 0:
            # 看预测值
            pred_box3d = pred_infos['box3d_lidar']
            proj_corners3d_list = pred_infos['proj_corners3d_list']
        else:
            # 看真值
            pred_box3d = gt_infos['box3d_lidar']
            proj_corners3d_list = gt_infos['proj_corners3d_list']
        pred_box3d_lidar = pred_box3d
        img_plus_box3d = paint_box3d_on_img(img.copy(), proj_corners3d_list, cls_idx=pred_box3d[:, 7],
                                            use_8_corners=False)
        publisher.publish_img(img_plus_box3d, self.img_plus_box3d_pub, self.bridge, stamp, self.frame_id)
        # marker.create_pred_detection_box3d_marker_array(pointcloud, pred_box3d_lidar.copy(),
        #                                                 score=None,
        #                                                 pc_color=None,
        #                                                 stamp=stamp,
        #                                                 frame_id=self.frame_id,
        #                                                 box3d_arr_marker=box3d_marker_array,
        #                                                 box3d_ns="pred/detection/box3d"
        #                                                 )

        # 跟踪真值数据
        if self.do_track:
            if 0:
                detections = gt_infos['box3d_camera']
            else:
                detections = pred_infos['box3d_camera']
            trackers = self.track(detections)
            trackers = object3d.box3d_from_kitti_to_lidar(trackers, self.cal_info)
            marker.create_pred_tracker_box3d_marker_array(pointcloud, trackers.copy(),
                                                          pc_color=None,
                                                          stamp=stamp,
                                                          frame_id=self.frame_id,
                                                          box3d_arr_marker=box3d_marker_array,
                                                          box3d_ns="pred/tracker/box3d",
                                                          tracker_text_ns="pred/tracker/text/tracker_id"
                                                          )
        colors_ros = color_o3d_to_color_ros(colors_np)
        pointcloud = np.column_stack((pointcloud, colors_ros))

        publisher.publish_pc_by_range(self.pc_in_pub,
                                      self.pc_out_pub,
                                      pointcloud,
                                      header,
                                      self.limit_range,
                                      field="xyzirgb")
        self.box3d_pub.publish(box3d_marker_array)
        # return mask

    def update_pred_infos(self, label_path, mask):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        idx_list = np.arange(mask.shape[0])[mask]
        # 仅保留指定行数
        selected_lines = [lines[i] for i in idx_list]

        # 写入新文件
        output_dir = Path("/home/helios/mnt/dataset/livox_dataset/training/update")
        output_file = output_dir / Path(label_path).name
        with open(output_file, 'w') as f:
            f.writelines(selected_lines)

    def calculate_iou(self, boxA, boxB):
        # 计算两个框的交集面积
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # 计算两个框的并集面积
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        union_area = boxA_area + boxB_area - intersection_area

        # 计算交占比
        iou = intersection_area / union_area

        return iou

    def filter_detetion_by_img_decision(self, multi_box2d_img, multi_box2d_lidar, iou_threshold=0.5):
        """
        对于特定类别使用图像的决策信息进行更新
        :param multi_box2d_img:
        :param multi_box2d_lidar:
        :param iou_threshold:
        :return:
        """
        mask = np.zeros(multi_box2d_lidar.shape[0], dtype=np.bool_)
        for i, box2d_lidar in enumerate(multi_box2d_lidar):
            for box2d_img in multi_box2d_img:
                if int(box2d_lidar[4]) == int(box2d_img[4]):
                    iou = self.calculate_iou(box2d_img[:4], box2d_lidar[:4])
                    if iou > iou_threshold:
                        mask[i] = True
                        break
        return mask

    def kitti_object_to_pcdet_object(self, label_path):
        """
        将KITTI的标签转换为PCDet的标签，并提取真值点云+二维框+三维角点信息
        :param label_path:
        :return:
        """
        obj_list = get_objects_from_label(label_path)

        if len(obj_list) == 0:
            return None

        # 此处的临时变量用于提高可读性
        annotations = {'name': np.array([obj.cls_type for obj in obj_list]),
                       'bbox': np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                       'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                       'location': np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0),
                       'rotation_y': np.array([obj.ry for obj in obj_list]),
                       'score': np.array([obj.score for obj in obj_list]),
                       'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                       'box2d': np.array([obj.box2d for obj in obj_list], np.int32),
                       'corners3d_cam': np.array([obj.corner3d_cam for obj in obj_list], np.float32)}

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])  # 可用的object数
        num_gt = len(annotations['name'])  # 标签中的object数
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        # note: Don't care都是在最后的，所以可以用这种方式处理
        loc = annotations['location'][:num_objects]  # x, y, z
        dims = annotations['dimensions'][:num_objects]  # l, w, h
        rots = annotations['rotation_y'][:num_objects]  # yaw
        score = annotations['score'][:num_objects]
        # points from camera frame->lidar frame
        loc_lidar = calibration.camera_to_lidar_points(loc, self.cal_info)
        class_name = annotations['name'][:num_objects]
        box2d = annotations['box2d'][:num_objects]
        corners3d_cam = annotations['corners3d_cam'][:num_objects]

        corners_3d_lidar = np.array([calibration.camera_to_lidar_points(corners3d, self.cal_info)
                                     for corners3d in corners3d_cam], np.float32)

        proj_corners2d_list = []
        proj_corners3d_list = []
        for corners3d in corners_3d_lidar:
            proj_corners2d, proj_corners3d = calibration.corner_3d8c_2d4c(corners3d, self.cal_info,
                                                                          frame="lidar")
            proj_corners2d_list.append(proj_corners2d)
            proj_corners3d_list.append(proj_corners3d)
        proj_corners2d_list = np.array(proj_corners2d_list, dtype=np.int32)
        proj_corners3d_list = np.array(proj_corners3d_list, dtype=np.int32)

        class_id = []
        for i in range(class_name.shape[0]):
            class_id.append(int(common_utils.cls_type_to_id(class_name[i])))
        class_id = np.asarray(class_id, dtype=np.int32).reshape(-1, 1)

        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2  # btn center->geometry center

        box3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]), class_id], axis=1)
        box3d_camera = np.concatenate([h, w, l, loc, rots[..., np.newaxis], class_id], axis=1)

        infos = {
            'box3d_lidar': box3d_lidar,
            'box3d_camera': box3d_camera,
            'box2d': box2d,
            'corners3d_cam': corners3d_cam,
            'proj_corners2d_list': proj_corners2d_list,
            'proj_corners3d_list': proj_corners3d_list,
            'score': score,
            'cls_id': class_id
        }
        return infos

    def get_kitti_obj(self, label_path):
        if not Path(label_path).exists():
            return None
        infos = self.kitti_object_to_pcdet_object(label_path)
        return infos

    def iter_dataset_with_kitti_label(self):

        label_paths = sorted(list(Path(self.label_dir_path).glob('*.txt')))
        start_label_id = 177
        for frame_id, gt_path in enumerate(
                tqdm(label_paths[start_label_id:],
                     initial=start_label_id, total=len(label_paths), desc='Loading dataset'),
                start=start_label_id):
            if __ROS_VERSION__ == 1 and rospy.is_shutdown():
                exit(0)
            if __ROS_VERSION__ == 2 and not rclpy.ok():
                exit(0)

            file_idx = gt_path.stem
            image_path = "{}/{}.jpg".format(self.image_dir_path, file_idx)
            lidar_path = "{}/{}.bin".format(self.pointcloud_dir_path, file_idx)
            # self.prediction_dir_path = "/home/helios/Github/workspace/livox_detection/OpenPCDet/output/livox_models/PartA2_free.kitti/default/eval/epoch_7872/val/default/final_result/data/"
            # self.prediction_dir_path = "/home/helios/Github/workspace/livox_detection/OpenPCDet/output/livox_models/pointpillar.kitti/default/eval/epoch_7728/val/default/final_result/data/"
            self.prediction_dir_path = "/home/helios/Github/workspace/livox_detection/OpenPCDet/output/livox_models/second/default/eval/epoch_7862/val/default/final_result/data/"

            pred_path = "{}/{}.txt".format(self.prediction_dir_path, file_idx)

            # 读取标签数据（真值和预测值）
            gt_infos = self.get_kitti_obj(gt_path)
            if gt_infos is None:
                continue
            # pred_infos = self.get_kitti_obj(pred_path)
            # if pred_infos is None:
            #     continue

            # 读取传感器数据
            image = cv2.imread(image_path)
            pointcloud = load_pointcloud(lidar_path)

            # 预测
            self.predict_and_publish(pointcloud, image, gt_infos, pred_infos=None)
            # self.update_pred_infos(pred_path, mask)

            if self.auto_update:
                time.sleep(self.update_time)
            else:
                input()


def save_single_prediction_in_kitti_format(single_pred_dict, calib, image_shape, output_path=None):
    """导出KITTI格式的预测结果
    :param single_pred_dict:
    :param output_path:
    :return:
    """

    pred_scores = single_pred_dict['pred_scores']
    pred_boxes = single_pred_dict['pred_boxes']
    pred_labels = single_pred_dict['pred_labels']

    pred_boxes_camera = object3d.box3d_from_lidar_to_kitti((pred_boxes, calib))
    pred_boxes_img = calibration.corner_3d8c_2d4c(
        pred_boxes_camera, calib, frame="camera", img_shape=image_shape
    )

    # single_pred_dict['name'] = np.array(class_names)[pred_labels - 1]
    single_pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
    single_pred_dict['bbox'] = pred_boxes_img
    single_pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
    single_pred_dict['location'] = pred_boxes_camera[:, 0:3]
    single_pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
    single_pred_dict['score'] = pred_scores
    single_pred_dict['boxes_lidar'] = pred_boxes

    if output_path is not None:
        cur_det_file = output_path
        with open(cur_det_file, 'w') as f:
            bbox = single_pred_dict['bbox']
            loc = single_pred_dict['location']
            dims = single_pred_dict['dimensions']  # lhw -> hwl

            for idx in range(len(bbox)):
                print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                      % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                         bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                         dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                         loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                         single_pred_dict['score'][idx]), file=f)


if __name__ == '__main__':
    if __ROS_VERSION__ == 1:
        Visualization(Node)
    elif __ROS_VERSION__ == 2:
        rclpy.init()
        node = Visualization()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

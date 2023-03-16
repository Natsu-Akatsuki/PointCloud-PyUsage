import os
import time
from pathlib import Path

import common_utils
import cv2
import numpy as np
from tqdm import tqdm

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

from cv_bridge import CvBridge
import std_msgs.msg
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
# isort: on

import ampcl.visualization
import common_utils
from ampcl.calibration.calibration_livox import LivoxCalibration
from ampcl.calibration.object3d_kitti import get_objects_from_label
from ampcl.io import load_pointcloud
from ampcl.ros_utils import np_to_pointcloud2
from ampcl.visualization import color_o3d_to_color_ros
from tqdm import tqdm


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Visualization(Node):
    def __init__(self):
        super().__init__("visualization")
        cfg = common_utils.Config("config/livox.yaml")

        # ROS
        if __ROS_VERSION__ == 1:
            self.pointcloud_pub = rospy.Publisher(cfg.pointcloud_topic, PointCloud2, queue_size=1, latch=True)
            self.img_pub = rospy.Publisher(cfg.image_topic, Image, queue_size=1, latch=True)
            self.bbx_pub = rospy.Publisher(cfg.bounding_box_topic, MarkerArray, queue_size=1, latch=True)
        if __ROS_VERSION__ == 2:
            latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.pointcloud_pub = self.create_publisher(PointCloud2, cfg.pointcloud_topic, latching_qos)
            self.img_pub = self.create_publisher(Image, cfg.image_topic, latching_qos)
            self.bbx_pub = self.create_publisher(MarkerArray, cfg.bounding_box_topic, latching_qos)

        self.frame = cfg.frame
        self.bridge = CvBridge()

        # Data
        dataset_dir = Path(cfg.dataset_dir)
        self.image_dir_path = dataset_dir / Path(cfg.image_dir_path)
        self.pointcloud_dir_path = dataset_dir / cfg.pointcloud_dir_path
        self.label_dir_path = dataset_dir / cfg.label_dir_path
        self.calib_dir_path = dataset_dir / cfg.calib_dir_path

        # Visualization
        self.auto_update = cfg.auto_update
        self.update_time = cfg.update_time

        self.iter_dataset_with_kitti_label()

    def publish_result(self, pointcloud, img, pred_bbxes=None, gt_bbxes=None):

        frame = self.frame
        if __ROS_VERSION__ == 1:
            stamp = rospy.Time.now()
        if __ROS_VERSION__ == 2:
            stamp = self.get_clock().now().to_msg()

        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame

        img_msg = self.bridge.cv2_to_imgmsg(cvim=img, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame

        # bbxes = np.vstack(pred_bbxes, gt_bbxes)
        bbxes = np.vstack(gt_bbxes)

        all_colors = common_utils.generate_colors()

        box_marker_array = MarkerArray()
        colors_np = np.full((pointcloud.shape[0], 3), np.array([0.8, 0.8, 0.8]))
        if bbxes.shape[0] > 0:
            empty_marker = Marker()
            empty_marker.action = Marker.DELETEALL
            box_marker_array.markers.append(empty_marker)
            for i in range(bbxes.shape[0]):
                box = bbxes[i]
                cls_id = int(box[7])
                box_color = common_utils.id_to_color(cls_id)
                box_marker = common_utils.create_bounding_box_marker(box, stamp, i, box_color, frame_id=self.frame)
                box_marker_array.markers.append(box_marker)
                if cls_id == 2:
                    model_marker = common_utils.create_bounding_box_model(box, stamp, i, frame_id=self.frame)
                    box_marker_array.markers.append(model_marker)

                # 追加颜色
                point_color = all_colors[cls_id % len(all_colors)]
                indices_points_inside = common_utils.get_indices_of_points_inside(pointcloud, box, margin=0.1)
                colors_np[indices_points_inside] = point_color

        colors_ros = color_o3d_to_color_ros(colors_np)
        pointcloud = np.column_stack((pointcloud, colors_ros))
        pointcloud_msg = np_to_pointcloud2(pointcloud, header, field="xyzirgb")

        self.bbx_pub.publish(box_marker_array)
        self.pointcloud_pub.publish(pointcloud_msg)
        self.img_pub.publish(img_msg)

    def kitti_object_to_pcdet_object(self, label_path, cal_path):
        obj_list = get_objects_from_label(label_path)
        calib = LivoxCalibration(cal_path)

        if len(obj_list) == 0:
            return None

        # 此处的临时变量用于提高可读性
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])  # 可用的object数
        num_gt = len(annotations['name'])  # 标签中的object数
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        # note: Don't care都是在最后的，所以可以用这种方式处理
        loc = annotations['location'][:num_objects]  # x, y, z
        dims = annotations['dimensions'][:num_objects]  # l, w, h
        rots = annotations['rotation_y'][:num_objects]  # yaw
        loc_lidar = calib.camera_to_lidar_points(loc)  # points from camera frame->lidar frame
        class_name = annotations['name'][:num_objects]
        gt_class_id = []
        for i in range(class_name.shape[0]):
            gt_class_id.append(int(cls_type_to_id(class_name[i])))
        gt_class_id = np.asarray(gt_class_id, dtype=np.int32).reshape(-1, 1) + 1

        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2  # btn center->geometry center
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]), gt_class_id], axis=1)
        return gt_boxes_lidar

    def iter_dataset_with_kitti_label(self):

        label_paths = sorted(list(Path(self.label_dir_path).glob('*.txt')))
        for frame_id, label_path in enumerate(tqdm(label_paths)):
            if __ROS_VERSION__ == 1 and rospy.is_shutdown():
                exit(0)
            if __ROS_VERSION__ == 2 and not rclpy.ok():
                exit(0)

            file_idx = label_path.stem

            image_path = "{}/{}.jpg".format(self.image_dir_path, file_idx)
            lidar_path = "{}/{}.bin".format(self.pointcloud_dir_path, file_idx)
            calib_path = "{}/{}.txt".format(self.calib_dir_path, "config")

            calib_path = Path(calib_path).resolve()
            gt_boxes_lidar = self.kitti_object_to_pcdet_object(label_path, calib_path)
            if gt_boxes_lidar is None:
                print(f" No object in this frame {file_idx}, skip it.")
                continue

            image = cv2.imread(image_path)
            pointcloud = load_pointcloud(lidar_path)
            self.publish_result(pointcloud, image, pred_bbxes=None, gt_bbxes=gt_boxes_lidar)

            if self.auto_update:
                time.sleep(self.update_time)
            else:
                input()


def ros1_wrapper():
    Visualization(Node)


def ros2_wrapper():
    rclpy.init()
    node = Visualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    if __ROS_VERSION__ == 1:
        ros1_wrapper()
    elif __ROS_VERSION__ == 2:
        ros2_wrapper()

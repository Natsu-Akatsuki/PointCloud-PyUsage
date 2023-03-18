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

import time
from pathlib import Path

import common_utils
import cv2
import numpy as np
from ampcl import filter
from ampcl.calibration import calibration_template
from ampcl.calibration.calibration_livox import LivoxCalibration
from ampcl.calibration.object3d_kitti import get_objects_from_label
from ampcl.io import load_pointcloud
from ampcl.ros import np_to_pointcloud2
from ampcl.ros.marker import create_marker
from ampcl.visualization import color_o3d_to_color_ros
from tqdm import tqdm
from ab3dmot import AB3DMOT


class Visualization(Node):
    def __init__(self):
        super().__init__("visualization")
        cfg = common_utils.Config("config/livox.yaml")

        # ROS
        if __ROS_VERSION__ == 1:
            self.pointcloud_pub = rospy.Publisher(cfg.pointcloud_topic, PointCloud2, queue_size=1, latch=True)
            self.img_plus_box2d_pub = rospy.Publisher(cfg.image_topic, Image, queue_size=1, latch=True)
            self.box3d_pub = rospy.Publisher(cfg.bounding_box_topic, MarkerArray, queue_size=1, latch=True)
        if __ROS_VERSION__ == 2:
            latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.vanilla_pointcloud_pub = self.create_publisher(PointCloud2, cfg.vanilla_pointcloud_topic, latching_qos)
            self.color_pointcloud_pub = self.create_publisher(PointCloud2, cfg.color_pointcloud_topic, latching_qos)
            self.img_plus_box2d_pub = self.create_publisher(Image, cfg.img_plus_box2d_topic, latching_qos)
            self.img_plus_box3d_pub = self.create_publisher(Image, cfg.img_plus_box3d_topic, latching_qos)
            self.box3d_pub = self.create_publisher(MarkerArray, cfg.box3d_topic, latching_qos)

        self.frame = cfg.frame
        self.bridge = CvBridge()

        # Data
        dataset_dir = Path(cfg.dataset_dir)
        self.image_dir_path = dataset_dir / Path(cfg.image_dir_path)
        self.pointcloud_dir_path = dataset_dir / cfg.pointcloud_dir_path
        self.label_dir_path = dataset_dir / cfg.label_dir_path
        self.prediction_dir_path = dataset_dir / cfg.prediction_dir_path
        self.calib_dir_path = dataset_dir / cfg.calib_dir_path

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
            color_pointcloud_in_image, mask = calibration_template.paint_pointcloud(pointcloud, img,
                                                                                    self.cal_info)
            # 将不能投影到图像上的点云的颜色设置为黑色
            pointcloud_not_in_image = pointcloud[~mask]
            rgb_arr = np.zeros((pointcloud_not_in_image.shape[0]), dtype=np.float32)
            pointcloud_not_in_image = np.hstack((pointcloud_not_in_image, rgb_arr[:, np.newaxis]))
            color_pointcloud = np.vstack((color_pointcloud_in_image, pointcloud_not_in_image))
            color_pointcloud_msg = np_to_pointcloud2(color_pointcloud, header, field="xyzirgb")
            self.color_pointcloud_pub.publish(color_pointcloud_msg)

    def paint_box2d(self, image, box2d, cls_indices=None):

        for i in range(box2d.shape[0]):
            box = box2d[i]
            if cls_indices is None:
                box_color = (0, 0, 255)
            else:
                box_color = (np.asarray(common_utils.id_to_color(cls_indices[i])) * 255)[::-1]
                box_color = tuple([int(x) for x in box_color])
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), box_color, 2)

        return image

    def paint_box3d(self, image, corners3d_cam, use_8_corners=True, cls_indices=None):
        """
        corners3d_cam: [N, 8]
                 4-------- 5
               /|         /|
              7 -------- 6 .
              | |        | |
              . 0 -------- 1
              |/         |/
              3 -------- 2
        """
        for i in range(corners3d_cam.shape[0]):
            if cls_indices is None:
                box_color = (0, 0, 255)
            else:
                box_color = (np.asarray(common_utils.id_to_color(cls_indices[i])) * 255)[::-1]
                box_color = tuple([int(x) for x in box_color])

            bbx_3d = corners3d_cam[i]
            if use_8_corners:
                # draw the 4 vertical lines
                cv2.line(image, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[2].astype(int)), tuple(bbx_3d[6].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[3].astype(int)), tuple(bbx_3d[7].astype(int)), box_color, 2)

                # draw the 4 horizontal lines
                cv2.line(image, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[1].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[2].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[2].astype(int)), tuple(bbx_3d[3].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[3].astype(int)), tuple(bbx_3d[0].astype(int)), box_color, 2)

                # draw the 4 horizontal lines
                cv2.line(image, tuple(bbx_3d[4].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[5].astype(int)), tuple(bbx_3d[6].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[6].astype(int)), tuple(bbx_3d[7].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[7].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)

                cv2.line(image, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
                cv2.line(image, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)
            else:
                x1, y1 = int(np.min(bbx_3d, axis=0)), int(np.min(bbx_3d, axis=1))
                x2, y2 = int(np.max(bbx_3d, axis=0)), int(np.max(bbx_3d, axis=1))
                image_shape = image.shape
                x1 = np.clip(x1, a_min=0, a_max=image_shape[1] - 1)
                y1 = np.clip(y1, a_min=0, a_max=image_shape[0] - 1)
                x2 = np.clip(x2, a_min=0, a_max=image_shape[1] - 1)
                y2 = np.clip(y2, a_min=0, a_max=image_shape[0] - 1)
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        return image

    def publish_dataset(self, pointcloud, img, gt_infos, prediction_infos):

        gt_boxes_lidar = gt_infos['boxes3d_lidar']
        gt_boxes2d = gt_infos['boxes2d']
        proj_corners3d_list = gt_infos['proj_corners3d_list']

        if __ROS_VERSION__ == 1:
            stamp = rospy.Time.now()
        if __ROS_VERSION__ == 2:
            stamp = self.get_clock().now().to_msg()

        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = self.frame

        # 图像+2D框
        img_plus_box2d = self.paint_box2d(img.copy(), gt_boxes2d, cls_indices=gt_boxes_lidar[:, 7])
        img_msg = self.bridge.cv2_to_imgmsg(cvim=img_plus_box2d, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame
        self.img_plus_box2d_pub.publish(img_msg)

        # 图像+投影三维框
        img_plus_box3d = self.paint_box3d(img.copy(), proj_corners3d_list, cls_indices=gt_boxes_lidar[:, 7])
        img_msg = self.bridge.cv2_to_imgmsg(cvim=img_plus_box3d, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame
        self.img_plus_box3d_pub.publish(img_msg)

        # 彩色点云
        self.pub_color_pointcloud_ros(header, pointcloud, img)

        # 跟踪
        # cat_trackers, _ = self.car_tracker.track(gt_boxes_lidar[gt_boxes_lidar[:, 7] == 1][:, :7])
        # pedestrian_trackers, _ = self.pedestrian_tracker.track(gt_boxes_lidar[gt_boxes_lidar[:, 7] == 2][:, :7])
        # cyclist_trackers, _ = self.cyclist_tracker.track(gt_boxes_lidar[gt_boxes_lidar[:, 7] == 3][:, :7])

        # 将每一类别的跟踪结果进行合并
        # trackers = np.concatenate((cat_trackers, pedestrian_trackers, cyclist_trackers), axis=0)

        # 点云+3D框
        boxes = gt_boxes_lidar
        box_marker_array = MarkerArray()
        colors_np = np.full((pointcloud.shape[0], 3), np.array([0.8, 0.8, 0.8]))
        if boxes.shape[0] > 0:
            empty_marker = Marker()
            empty_marker.action = Marker.DELETEALL
            box_marker_array.markers.append(empty_marker)
            for i in range(boxes.shape[0]):
                box = boxes[i]
                cls_id = int(box[7])
                box_color = common_utils.id_to_color(cls_id)

                indices_points_inside = filter.get_indices_of_points_inside(pointcloud, box, margin=0.1)
                marker_dict = create_marker.create_bounding_box_marker(
                    box, stamp, frame_id=self.frame,
                    box_ns="shape", box_color=box_color, box_id=i,
                    track_ns="track", track_id=None,
                    confidence_ns="confidence", confidence=None)

                if indices_points_inside.shape[0] < 10:
                    box_color = np.array([1.0, 0.0, 0.0])
                colors_np[indices_points_inside] = box_color

                for marker_element in marker_dict.values():
                    box_marker_array.markers.append(marker_element)

        colors_ros = color_o3d_to_color_ros(colors_np)
        pointcloud = np.column_stack((pointcloud, colors_ros))
        pointcloud_msg = np_to_pointcloud2(pointcloud, header, field="xyzirgb")
        self.vanilla_pointcloud_pub.publish(pointcloud_msg)
        self.box3d_pub.publish(box_marker_array)

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
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
        annotations['box2d'] = np.array([obj.box2d for obj in obj_list], np.int32)
        annotations['corners3d_cam'] = np.array([obj.corners3d_cam for obj in obj_list], np.float32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])  # 可用的object数
        num_gt = len(annotations['name'])  # 标签中的object数
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        # note: Don't care都是在最后的，所以可以用这种方式处理
        loc = annotations['location'][:num_objects]  # x, y, z
        dims = annotations['dimensions'][:num_objects]  # l, w, h
        rots = annotations['rotation_y'][:num_objects]  # yaw
        # points from camera frame->lidar frame
        loc_lidar = calibration_template.camera_to_lidar_points(loc, self.cal_info)
        class_name = annotations['name'][:num_objects]
        boxes2d = annotations['box2d'][:num_objects]
        corners3d_cam = annotations['corners3d_cam'][:num_objects]

        corners_3d_lidar = np.array([calibration_template.camera_to_lidar_points(corners3d, self.cal_info)
                                     for corners3d in corners3d_cam], np.float32)

        proj_corners2d_list = []
        proj_corners3d_list = []
        for corners3d in corners_3d_lidar:
            proj_corners2d, proj_corners3d = calibration_template.corners_3d_to_2d(corners3d, self.cal_info)
            proj_corners2d_list.append(proj_corners2d)
            proj_corners3d_list.append(proj_corners3d)
        proj_corners2d_list = np.array(proj_corners2d_list, dtype=np.int32)
        proj_corners3d_list = np.array(proj_corners3d_list, dtype=np.int32)

        gt_class_id = []
        for i in range(class_name.shape[0]):
            gt_class_id.append(int(common_utils.cls_type_to_id(class_name[i])))
        gt_class_id = np.asarray(gt_class_id, dtype=np.int32).reshape(-1, 1)

        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2  # btn center->geometry center
        boxes3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]), gt_class_id], axis=1)

        infos = {
            'boxes3d_lidar': boxes3d_lidar,
            'boxes2d': boxes2d,
            'corners3d_cam': corners3d_cam,
            'proj_corners2d_list': proj_corners2d_list,
            'proj_corners3d_list': proj_corners3d_list
        }
        return infos

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
            prediction_path = "{}/{}.txt".format(self.prediction_dir_path, file_idx)

            prediction_infos = None
            if Path(self.prediction_dir_path).exists():
                prediction_infos = self.kitti_object_to_pcdet_object(prediction_path)
            gt_infos = self.kitti_object_to_pcdet_object(label_path)

            if gt_infos is None:
                print(f" No object in this frame {file_idx}, skip it.")
                continue

            image = cv2.imread(image_path)
            pointcloud = load_pointcloud(lidar_path)
            self.publish_dataset(pointcloud, image, gt_infos, prediction_infos)

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

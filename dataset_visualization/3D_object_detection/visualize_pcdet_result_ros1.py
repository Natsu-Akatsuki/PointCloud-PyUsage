import time
from pathlib import Path

import common_utils
import cv2
import numpy as np
import rospy
import std_msgs.msg
from common_utils import clear_bounding_box_marker
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray

from ampcl.ros_utils import np_to_pointcloud2
from ampcl.visualization import color_o3d_to_color_ros


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Visualization():
    def __init__(self):
        rospy.init_node("visualization")
        cfg = common_utils.Config("config/kitti.yaml")

        # ROS
        self.pointcloud_pub = rospy.Publisher(cfg.pointcloud_topic, PointCloud2, queue_size=1, latch=True)
        self.img_pub = rospy.Publisher(cfg.image_topic, Image, queue_size=1, latch=True)
        self.bbx_pub = rospy.Publisher(cfg.bounding_box_topic, MarkerArray, queue_size=5, latch=True)
        self.frame = cfg.frame
        self.bridge = CvBridge()

        # Data
        dataset_dir = Path(cfg.dataset_dir)
        gt_file = Path(cfg.gt_file).resolve()
        predict_file = Path(cfg.predict_file).resolve()
        self.gt_infos = common_utils.open_pkl_file(gt_file)
        self.pred_infos = common_utils.open_pkl_file(predict_file)
        self.image_dir_path = str(dataset_dir / "training/image_2")
        self.lidar_dir_path = str(dataset_dir / "training/velodyne")

        # Visualization
        self.auto_update = cfg.auto_update
        self.update_time = cfg.update_time

        self.last_box_num = 0
        self.iter_dataset()

    def publish_result(self, pointcloud, img, pred_bbxes=None, gt_bbxes=None):

        frame = "lidar"
        stamp = rospy.Time.now()
        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame
        pointcloud_msg = np_to_pointcloud2(pointcloud, header)

        img_msg = self.bridge.cv2_to_imgmsg(cvim=img, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame

        # bbxes = np.vstack(pred_bbxes, gt_bbxes)
        bbxes = np.vstack(gt_bbxes)

        all_colors = common_utils.generate_colors()

        box_marker_array = MarkerArray()
        colors_np = np.full((pointcloud.shape[0], 3), np.array([0.8, 0.8, 0.8]))
        if bbxes.shape[0] > 0:
            for i in range(bbxes.shape[0]):
                box = bbxes[i]
                cls_id = int(box[7])
                box_color = common_utils.id_to_color(cls_id)
                box_marker = common_utils.create_bounding_box_marker(box, stamp, i, box_color, frame_id="lidar")
                box_marker_array.markers.append(box_marker)
                if cls_id == 2:
                    model_marker = common_utils.create_bounding_box_model(box, stamp, i, frame_id="lidar")
                    box_marker_array.markers.append(model_marker)

                # 追加颜色
                point_color = all_colors[cls_id % len(all_colors)]
                indices_points_inside = common_utils.get_indices_of_points_inside(pointcloud, box)
                colors_np[indices_points_inside] = point_color

        # clear history marker （RViz中的marker是追加的，取非替换或设置显示时间否则会残留）
        if self.last_box_num > 0:
            empty_marker_array = MarkerArray()
            for i in range(self.last_box_num):
                empty_shape_marker = clear_bounding_box_marker(stamp, i, ns="shape", frame_id="lidar")
                empty_marker_array.markers.append(empty_shape_marker)
                empty_model_marker = clear_bounding_box_marker(stamp, i, ns="model", frame_id="lidar")
                empty_marker_array.markers.append(empty_model_marker)
            self.bbx_pub.publish(empty_marker_array)
        self.last_box_num = bbxes.shape[0]

        colors_ros = color_o3d_to_color_ros(colors_np)
        pointcloud = np.column_stack((pointcloud, colors_ros))
        pointcloud_msg = np_to_pointcloud2(pointcloud, header, field="xyzirgb")

        self.bbx_pub.publish(box_marker_array)
        self.pointcloud_pub.publish(pointcloud_msg)
        self.img_pub.publish(img_msg)

    def iter_dataset(self):

        for i in range(len(self.gt_infos)):
            if rospy.is_shutdown():
                exit(0)

            file_idx = self.gt_infos[i]['point_cloud']['lidar_idx']
            image_path = "{}/{}.png".format(self.image_dir_path, file_idx)
            lidar_path = "{}/{}.bin".format(self.lidar_dir_path, file_idx)

            gt_boxes_lidar = self.gt_infos[i]["annos"]["gt_boxes_lidar"]
            gt_num = gt_boxes_lidar.shape[0]
            gt_boxes_name = self.gt_infos[i]["annos"]["name"]

            gt_class_id = []
            for j in range(gt_num):
                gt_class_id.append(int(cls_type_to_id(gt_boxes_name[j])))
            gt_class_id = np.asarray(gt_class_id, dtype=np.int32).reshape(-1, 1) + 1
            gt_boxes_lidar = np.concatenate((gt_boxes_lidar, gt_class_id, np.ones_like(gt_class_id)), axis=-1)

            image = cv2.imread(image_path)
            pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

            pred_info = self.pred_infos[i]
            pred_info_id = []
            for j in range(pred_info["name"].shape[0]):
                pred_info_id.append(int(cls_type_to_id(pred_info["name"][j])))
            pred_class_id = np.asarray(pred_info_id, dtype=np.int32) + 1 + 4

            pred_boxes_lidar = np.concatenate((pred_info["boxes_lidar"],
                                               pred_class_id.reshape(-1, 1),
                                               pred_info["score"].reshape(-1, 1)), axis=-1)
            self.publish_result(pointcloud, image, pred_bbxes=pred_boxes_lidar, gt_bbxes=gt_boxes_lidar)

            if self.auto_update:
                time.sleep(self.update_time)
            else:
                input()


if __name__ == '__main__':
    node = Visualization()

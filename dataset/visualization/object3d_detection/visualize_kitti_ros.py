import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ampcl.calibration import object3d_kitti, calibration
from ampcl.calibration.calibration_kitti import KITTICalibration
from ampcl.io import load_pointcloud
from ampcl.perception import ground_segmentation_ransac
from ampcl.ros import marker, publisher
from ampcl.visualization import color_o3d_to_color_ros, paint_box2d_on_img
from easydict import EasyDict
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

from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray


# isort: on


class Visualization(Node):
    def __init__(self):
        super().__init__("visualization")
        self.init_param()
        self.load_dataset()

    def init_param(self):
        cfg_file = "config/kitti.yaml"
        with open(cfg_file, 'r') as f:
            cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

        self.limit_range = cfg.AlgorithmParam.limit_range
        self.frame_id = cfg.ROSParam.frame_id

        if __ROS_VERSION__ == 1:
            # TODO
            pass
        if __ROS_VERSION__ == 2:
            latching_qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

            self.pub_dict = {
                # 真值框（相机二维框，激光雷达四角点二维框，激光雷达八角点二维框）
                "/gt/img/box2d4c_cam": self.create_publisher(Image, "/gt/img/box2d4c_cam", latching_qos),
                "/gt/img/box2d4c_lidar": self.create_publisher(Image, "/gt/img/box2d4c_lidar", latching_qos),
                "/gt/img/box2d8c_lidar": self.create_publisher(Image, "/gt/img/box2d8c_lidar", latching_qos),

                # 预测框（相机二维框，激光雷达四角点二维框（融合前和融合后），激光雷达八角点二维框）
                "/pred/img/box2d4c_cam": self.create_publisher(Image, "/pred/img/box2d4c_cam", latching_qos),
                "/pred/img/box2d_lidar": self.create_publisher(Image, "/pred/img/box2d_lidar", latching_qos),
                "/pred/img/box2d4c_lidar2": self.create_publisher(Image, "/pred/img/box2d4c_lidar2", latching_qos),

                # marker
                "/box3d_marker": self.create_publisher(MarkerArray, "/box3d_marker", latching_qos),
                "/distance_marker": self.create_publisher(MarkerArray, "/distance_marker", latching_qos),

                # pc（区域内的点云和区域外的点云）
                "/pc/in_region": self.create_publisher(PointCloud2, "/pc/in_region", latching_qos),
                "/pc/out_region": self.create_publisher(PointCloud2, "/pc/out_region", latching_qos)
            }

            distance_marker = marker.create_distance_marker(frame_id=self.frame_id, distance_delta=10)
            self.pub_dict["/distance_marker"].publish(distance_marker)

            # Algorithm
            self.auto_update = cfg.AlgorithmParam.auto_update
            self.update_time = cfg.AlgorithmParam.update_time
            self.apply_fov_filter = cfg.AlgorithmParam.apply_fov_filter

            # Data
            dataset_dir = Path(cfg.DatasetParam.dataset_dir)
            self.img_dir = dataset_dir / Path(cfg.DatasetParam.img_dir)
            self.pc_dir = dataset_dir / cfg.DatasetParam.pc_dir
            self.gt_label_dir = dataset_dir / cfg.DatasetParam.gt_label_dir
            self.cal_dir = dataset_dir / cfg.DatasetParam.cal_dir

    def load_dataset(self):

        split_file = "/home/helios/mnt/dataset/Kitti/object/ImageSets/train.txt"
        with open(split_file, 'r') as f:
            lines = f.readlines()

        pbar = tqdm(lines)
        for i, file_idx in enumerate(pbar):
            if (__ROS_VERSION__ == 1 and rospy.is_shutdown()) \
                    or (__ROS_VERSION__ == 2 and not rclpy.ok()):
                exit(0)

            # 读取图片、点云、标定外参数据
            file_idx = file_idx.strip()
            gt_label_path = "{}/{}.txt".format(self.gt_label_dir, file_idx)
            img_path = "{}/{}.png".format(self.img_dir, file_idx)
            pc_path = "{}/{}.bin".format(self.pc_dir, file_idx)
            cal_path = "{}/{}.txt".format(self.cal_dir, file_idx)

            calib = KITTICalibration(cal_path)
            cal_info = calib.cal_info

            gt_info = object3d_kitti.kitti_object_to_pcdet_object(gt_label_path, cal_info)
            img = cv2.imread(img_path)
            pc_np = load_pointcloud(pc_path)

            self.publish_result(pc_np, img, gt_info, cal_info)
            pbar.set_description("Finish frame %s" % file_idx)
            if self.auto_update:
                time.sleep(self.update_time)
            else:
                input()

    def publish_result(self, pc_np, img, gt_info, cal_info):

        stamp = rospy.Time.now() if __ROS_VERSION__ == 1 else self.get_clock().now().to_msg()
        header = publisher.create_header(stamp, self.frame_id)

        box3d_marker_array = marker.init_marker_array()

        # 只保留在相机视野内的点云
        if self.apply_fov_filter:
            _, _, mask = calibration.lidar_to_pixel(pc_np, cal_info, img_shape=img.shape[:2], use_mask=True)
            pc_np = pc_np[mask]

        pc_color = np.full((pc_np.shape[0], 3), np.array([0.8, 0.8, 0.8]))
        # 地面分割
        if 1:
            limit_range = (0, 50, -10, 10, -2.5, -1.5)  # KITTI数据集
            plane_model, ground_mask = ground_segmentation_ransac(pc_np, limit_range, distance_threshold=0.1,
                                                                  debug=False)
            pc_color[ground_mask] = [0.61, 0.46, 0.33]

        # 聚类分割：
        # todo

        if gt_info is not None:
            # Debug: 发布图像2D真值框
            gt_box2d4c = gt_info['box2d']
            gt_box3d_lidar = gt_info['box3d_lidar']
            img_debug = paint_box2d_on_img(img.copy(), gt_box2d4c, cls_idx=gt_box2d4c[:, 4])
            publisher.publish_img(img_debug, self.pub_dict["/gt/img/box2d4c_cam"], stamp, self.frame_id)
            marker.create_box3d_marker_array(box3d_marker_array, gt_box3d_lidar, stamp, self.frame_id,
                                             color_method="class",
                                             pc_np=pc_np, pc_color=pc_color,
                                             box3d_ns="gt/detection/box3d")

        colors_ros = color_o3d_to_color_ros(pc_color)
        pc_infer = np.column_stack((pc_np, colors_ros))
        publisher.publish_pc_by_range(self.pub_dict["/pc/in_region"],
                                      self.pub_dict["/pc/out_region"],
                                      pc_infer,
                                      header,
                                      self.limit_range,
                                      field="xyzirgb",
                                      )
        self.pub_dict["/box3d_marker"].publish(box3d_marker_array)


if __name__ == '__main__':
    if __ROS_VERSION__ == 1:
        Visualization(Node)
    elif __ROS_VERSION__ == 2:
        rclpy.init()
        node = Visualization()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

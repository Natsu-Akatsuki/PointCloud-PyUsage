import os
from pathlib import Path

import cv2
import numpy as np
import rospy
import std_msgs.msg
from cv_bridge import CvBridge
from dataset_utils.calibration_livox import Calibration
from matplotlib import pyplot as plt
from ros_numpy.point_cloud2 import (xyzi_numpy_to_pointcloud2,
                                    xyzirgb_numpy_to_pointcloud2)
from sensor_msgs.msg import Image, PointCloud2
from tqdm import tqdm


def bgr_to_hex(color_np):
    """
    Args:
        color_np:{n,3} [b,g,r] (opencv)->[r,g,b] (rviz)
    """
    # b = color_np[:, 0]
    # g = color_np[:, 1]
    # r = color_np[:, 2]

    rgb_channel = np.array((color_np[:, 2] << 16) | (color_np[:, 1] << 8) | \
                           (color_np[:, 0] << 0), dtype=np.uint32)

    rgb_channel = rgb_channel.view(np.float32)
    return rgb_channel


class LivoxDataset():
    def __init__(self, cal_path):
        rospy.init_node('kitti_dataset', anonymous=False)

        self.cal = Calibration(cal_path)
        self.pointcloud_pub = rospy.Publisher("/kitti_dataset/pointcloud", PointCloud2, queue_size=1)
        self.img_pub = rospy.Publisher("/kitti_dataset/image", Image, queue_size=1)

        self.bridge = CvBridge()
        self.stamp = rospy.Time.now()

        # pointcloud range
        self.limit_range = [0, -40, -3, 70, 50, 1]

    def mask_points_by_range(self, points, limit_range):
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
               & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
               & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
        return mask

    def mask_outliers(self, pointcloud):
        """
        mask dead region and outliers
        Args:
            pointcloud:
        """
        mask = ~((pointcloud[:, 0] == 0) & (pointcloud[:, 1] == 0) & (pointcloud[:, 2] == 0) | (
                (np.abs(pointcloud[:, 0]) < 2.0) & (np.abs(pointcloud[:, 1] < 1.5))))
        return mask

    def paint_pointcloud(self, pointcloud, image):
        # step1: passthrough filter
        pointcloud = pointcloud[self.mask_points_by_range(pointcloud, self.limit_range)]
        pointcloud = pointcloud[self.mask_outliers(pointcloud)][:, :4]

        # step2: project the point onto image
        proj_img, depth_camera = self.cal.lidar_to_img(pointcloud[:, :3])

        # undistort the image
        image = cv2.undistort(src=image, cameraMatrix=self.cal.intri_matrix, distCoeffs=self.cal.distor)

        # step3: remove the points which are out of the image
        H = image.shape[0]
        W = image.shape[1]
        point_in_image_mask = (proj_img[:, 0] >= 0) & (proj_img[:, 0] < W) & \
                              (proj_img[:, 1] >= 0) & (proj_img[:, 1] < H)

        proj_img = proj_img[point_in_image_mask]
        pointcloud = pointcloud[point_in_image_mask]

        # step4: index the color of the points
        bgr_channels = image[np.int_(proj_img[:, 1]), np.int_(proj_img[:, 0])].astype(np.uint32)  # H, W
        hex_channels = bgr_to_hex(bgr_channels)
        pointcloud = np.concatenate((pointcloud[:, :4], hex_channels.reshape(-1, 1)), axis=-1)
        self.publish_pointcloud(pointcloud, color=True)
        self.publish_img(image)

    def publish_pointcloud(self, pointcloud_np, color=False):
        header = std_msgs.msg.Header()
        header.stamp = self.stamp
        header.frame_id = "livox_frame"

        if color:
            pointcloud_msg = xyzirgb_numpy_to_pointcloud2(pointcloud_np, header)
        else:
            pointcloud_msg = xyzi_numpy_to_pointcloud2(pointcloud_np, header)
        self.pointcloud_pub.publish(pointcloud_msg)

    def publish_img(self, img_np):
        img_msg = self.bridge.cv2_to_imgmsg(cvim=img_np, encoding="passthrough")
        img_msg.header.stamp = self.stamp
        img_msg.header.frame_id = "livox"
        self.img_pub.publish(img_msg)


if __name__ == '__main__':
    cal_cfg = "config.txt"
    livox_dataset = LivoxDataset(cal_cfg)
    dataset_path = Path("kitti_dataset/training")

    r = rospy.Rate(10)
    while not rospy.is_shutdown():

        image_dir_path = str(dataset_path / "image")
        lidar_dir_path = str(dataset_path / "lidar")
        images = sorted(os.listdir(image_dir_path))
        lidars = sorted(os.listdir(lidar_dir_path))

        if len(images) != len(lidars):
            print("Number of image files != number of pcd files, please check")
            exit(1)

        for idx in tqdm(range(len(images))):
            if rospy.is_shutdown():
                break
            image_path = "{}/{:0>6d}.jpg".format(image_dir_path, idx)
            lidar_path = "{}/{:0>6d}.bin".format(lidar_dir_path, idx)
            image = cv2.imread(image_path)
            pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            livox_dataset.stamp = rospy.Time.now()
            livox_dataset.paint_pointcloud(pointcloud, image)
            print(f"current is {idx}")
        r.sleep()

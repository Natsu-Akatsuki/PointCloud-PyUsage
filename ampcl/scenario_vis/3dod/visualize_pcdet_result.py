import pickle
from pathlib import Path

import cv2
import numpy as np
import rospy
import std_msgs.msg
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from ros_numpy.point_cloud2 import xyzi_numpy_to_pointcloud2
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2
from tqdm import trange


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class KittiDataset():
    def __init__(self):
        rospy.init_node('kitti_dataset', anonymous=False)

        self.pointcloud_pub = rospy.Publisher("/pcdet/kitti/reduce/pointcloud", PointCloud2, queue_size=1)
        self.img_pub = rospy.Publisher("/pcdet/kitti/image", Image, queue_size=1)
        self.bbx_pub = rospy.Publisher('/pcdet/kitti/gt_bbx', BoundingBoxArray, queue_size=1)
        self.bridge = CvBridge()

    def publish_result(self, pointcloud, img, pred_bbxes=None, gt_bbxes=None):
        """
        visualize pointcloud, bounding box and its corresponding image
        :param pointcloud:
        :param img:
        :param pred_bbxes:
        :param bbxes:
        """
        frame = "kitti"
        stamp = rospy.Time.now()

        header = std_msgs.msg.Header()
        header.stamp = stamp
        header.frame_id = frame
        pointcloud_msg = xyzi_numpy_to_pointcloud2(pointcloud, header)

        img_msg = self.bridge.cv2_to_imgmsg(cvim=img, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame

        bbx_array = BoundingBoxArray()

        if isinstance(pred_bbxes, np.ndarray):
            bbxes = np.vstack((gt_bbxes, pred_bbxes))

        if bbxes.shape[0] > 0:
            for i in range(bbxes.shape[0]):
                bbx = bbxes[i]

                ros_bbx = BoundingBox()
                ros_bbx.header.frame_id = header.frame_id
                ros_bbx.header.stamp = header.stamp

                ros_bbx.pose.position.x = float(bbx[0])
                ros_bbx.pose.position.y = float(bbx[1])
                ros_bbx.pose.position.z = float(bbx[2])

                ros_bbx.dimensions.x = float(bbx[3])
                ros_bbx.dimensions.y = float(bbx[4])
                ros_bbx.dimensions.z = float(bbx[5])

                # rotation = Rotation.from_rotvec([0, 0, bbx[6] + 1e-6])
                rotation = Rotation.from_euler("ZYX", [float(bbx[6]), 0, 0])
                quat = rotation.as_quat()
                ros_bbx.pose.orientation.x = quat[0]
                ros_bbx.pose.orientation.y = quat[1]
                ros_bbx.pose.orientation.z = quat[2]
                ros_bbx.pose.orientation.w = quat[3]

                ros_bbx.label = int(bbx[7])
                ros_bbx.value = float(bbx[8])
                bbx_array.boxes.append(ros_bbx)

        bbx_array.header.stamp = stamp
        bbx_array.header.frame_id = frame

        self.bbx_pub.publish(bbx_array)
        self.pointcloud_pub.publish(pointcloud_msg)
        self.img_pub.publish(img_msg)


def open_pkl_file(infos_path):
    with open(str(infos_path), 'rb') as f:
        infos = pickle.load(f)
    return infos


if __name__ == '__main__':

    kitti_dataset = KittiDataset()
    dataset_dir = Path("/mnt/disk2/dataset/Kitti/object/")

    gt_infos_path = Path("data/kitti_infos_val.pkl").resolve()
    result_infos_path = Path("data/result.pkl").resolve()
    gt_infos = open_pkl_file(gt_infos_path)
    pred_infos = open_pkl_file(result_infos_path)

    image_dir_path = str(dataset_dir / "training/image_2")
    lidar_dir_path = str(dataset_dir / "training/velodyne_reduced")

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        for i in trange(len(gt_infos)):
            file_idx = gt_infos[i]['point_cloud']['lidar_idx']
            image_path = "{}/{}.png".format(image_dir_path, file_idx)
            lidar_path = "{}/{}.bin".format(lidar_dir_path, file_idx)

            if rospy.is_shutdown():
                break
            gt_boxes_lidar = gt_infos[i]["annos"]["gt_boxes_lidar"]
            gt_num = gt_boxes_lidar.shape[0]
            gt_boxes_name = gt_infos[i]["annos"]["name"]

            gt_class_id = []
            for j in range(gt_num):
                gt_class_id.append(int(cls_type_to_id(gt_boxes_name[j])))
            gt_class_id = np.asarray(gt_class_id, dtype=np.int32).reshape(-1, 1) + 1

            gt_boxes_lidar = np.concatenate((gt_boxes_lidar, gt_class_id, np.ones_like(gt_class_id)), axis=-1)
            image = cv2.imread(image_path)
            pointcloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

            pred_info = pred_infos[i]
            pred_info_id = []
            for j in range(pred_info["name"].shape[0]):
                pred_info_id.append(int(cls_type_to_id(pred_info["name"][j])))
            pred_class_id = np.asarray(pred_info_id, dtype=np.int32) + 1 + 4

            pred_boxes_lidar = np.concatenate((pred_info["boxes_lidar"],
                                               pred_class_id.reshape(-1, 1),
                                               pred_info["score"].reshape(-1, 1)), axis=-1)
            kitti_dataset.publish_result(pointcloud, image, pred_bbxes=pred_boxes_lidar, gt_bbxes=gt_boxes_lidar)
            r.sleep()

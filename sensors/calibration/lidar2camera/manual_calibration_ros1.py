import cv2
import numpy as np
import rospy
from ampcl.calibration.transform import ros_frame_tf_to_basic_change
from ampcl.filter import c_voxel_filter, passthrough_filter
from ampcl.ros_utils import (DDynamicReconfigure, pointcloud2_to_xyzi_array,
                             xyzrgb_numpy_to_pointcloud2)
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from ampcl.io import load_pointcloud
from cal_experiment import QualExperiment
from common import get_calib_from_file


class ManualCalibration():
    def __init__(self, mode=2):

        self.mode = mode
        # 读取内参和外参，并进行动态参数的配置
        self.is_param_init = False
        param_file = "calib_parameter.txt"
        calib = get_calib_from_file(param_file)
        self.intri_matrix = calib['intri_matrix']
        self.distor = calib['distor']  # 0.2, 0, -0.18, 88, 0, -90
        self.xyz_ypr = calib["xyz_ypr"]
        self.extri_matrix = ros_frame_tf_to_basic_change(self.xyz_ypr)
        self.ddynrec = DDynamicReconfigure("ManualCalibration")
        self.dyn_reconfigure()

        with np.printoptions(precision=2, suppress=True):
            print("初始相机外参（【坐标变换】将激光雷达系的点转换到相机系下）：")
            print(self.extri_matrix)

        # 配置定量分析
        self.qual_experiment = QualExperiment()
        self.qual_experiment.distor = self.distor
        self.qual_experiment.intri_matrix = self.intri_matrix
        self.qual_experiment.extri_matrix = self.extri_matrix

        self.pc_pub = rospy.Publisher("/cal_pointcloud", PointCloud2, queue_size=10, latch=True)
        self.img_pub = rospy.Publisher("/cal_img", Image, queue_size=1, latch=True)

        self.bridge = CvBridge()
        self.img = None
        self.pointcloud_np = None

        # 基于更新的外参，发布动态TF
        self.tf_broadcaster = TransformBroadcaster()

    def on_subscriber(self):
        lidar_topic_name = "/sensing/lidar/rslidar_points"
        img_topic_name = "/camera/raw_img"
        self.pc_sub = rospy.Subscriber(PointCloud2, lidar_topic_name, self.pc_callback, queue_size=10)
        self.img_sub = rospy.Subscriber(Image, img_topic_name, self.img_callback, queue_size=10)

    def dyn_reconfigure(self):
        self.ddynrec.add_variable("tx", "float/double variable", self.xyz_ypr[0], -1.0, 1.0)
        self.ddynrec.add_variable("ty", "float/double variable", self.xyz_ypr[1], -1.0, 1.0)
        self.ddynrec.add_variable("tz", "float/double variable", self.xyz_ypr[2], -1.0, 1.0)
        self.ddynrec.add_variable("yaw", "float/double variable", self.xyz_ypr[3], -180, 180)
        self.ddynrec.add_variable("pitch", "float/double variable", self.xyz_ypr[4], -180, 180)
        self.ddynrec.add_variable("roll", "float/double variable", self.xyz_ypr[5], -180, 180)

        params_dict = {
            "tx": self.xyz_ypr[0],
            "ty": self.xyz_ypr[1],
            "tz": self.xyz_ypr[2],
            "yaw": self.xyz_ypr[3],
            "pitch": self.xyz_ypr[4],
            "roll": self.xyz_ypr[5],
        }
        for key, value in params_dict.items():
            self.__setattr__(key, value)

        # Start the server
        self.ddynrec.start(self.param_callback)

    def param_callback(self, config, level):
        if self.is_param_init == False:
            self.is_param_init = True
            return config

        # Update all variables
        params = self.ddynrec.get_variable_names()
        for param in params:
            self.__dict__[param] = config[param]

        xyz_ypr = np.array([self.tx, self.ty, self.tz, self.yaw, self.pitch, self.roll])
        self.qual_experiment.extri_matrix = ros_frame_tf_to_basic_change(xyz_ypr)

        self.fuse_data(debug=False)
        self.publish_fused_data()

        return config

    def publish_tranform(self):
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "lidar"
        t.child_frame_id = "camera"

        t.transform.translation.x = self.xyz_ypr[0]
        t.transform.translation.y = self.xyz_ypr[1]
        t.transform.translation.z = self.xyz_ypr[2]

        yaw = self.xyz_ypr[3]
        pitch = self.xyz_ypr[4]
        roll = self.xyz_ypr[5]
        q = Rotation.from_euler('ZYX', (yaw, pitch, roll), degrees=True).as_quat()

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def pc_callback(self, pc_msg):
        self.pointcloud_np = pointcloud2_to_xyzi_array(pc_msg)

    def img_callback(self, img_msg):
        """
        preprocess snippet:
        self.img = cv2.flip(self.img, 1)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        :param img_msg:
        :return:
        """
        self.img = self.bridge.imgmsg_to_cv2(img_msg)

    def fuse_data(self, debug=False):
        self.fused_pointcloud = self.qual_experiment.paint_pointcloud(self.pointcloud_np[:, :4], self.img, debug=debug)
        self.fused_img = self.qual_experiment.project_pointcloud_to_img(self.pointcloud_np[:, :4], self.img,
                                                                        fields=("intensity", "depth"),
                                                                        debug=debug)

    def publish_fused_data(self):

        img_ros = self.bridge.cv2_to_imgmsg(self.fused_img)
        self.img_pub.publish(img_ros)

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "lidar"
        pointcloud_ros = xyzrgb_numpy_to_pointcloud2(self.fused_pointcloud, header)
        self.pc_pub.publish(pointcloud_ros)

    def spin(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.publish_tranform()
            if self.pointcloud_np is not None and self.img is not None and self.mode == 2:
                self.fuse_data()
                self.publish_fused_data()
            r.sleep()


if __name__ == '__main__':
    rospy.init_node("manual_calibration")

    mode = 1
    manual_calibration = ManualCalibration(mode)

    pointcloud_file = "data/calibration.pcd"
    img_file = "data/calibration.jpeg"

    """
    mode0：读取点云文件和图片，用Open3D进行可视化
    mode1：读取点云文件和图片，用RVIZ进行可视化，可动态调参，定时发布
    mode2: 读取ROS点云数据和图片，用RVIZ进行可视化，可动态调参，定时发布
    """

    if mode == 0 or mode == 1:
        pointcloud_np = load_pointcloud(pointcloud_file)[:, :4]
        pointcloud_np = pointcloud_np[passthrough_filter(pointcloud_np, (0, 50, -40, 40, -2, 3))]
        pointcloud_np = c_voxel_filter(pointcloud_np, voxel_size=(0.02, 0.02, 0.02))
        manual_calibration.pointcloud_np = pointcloud_np
        manual_calibration.img = cv2.imread(img_file)
        if mode == 0:
            manual_calibration.fuse_data(debug=True)
            exit(0)
        if mode == 1:
            manual_calibration.fuse_data(debug=False)
            manual_calibration.publish_fused_data()
            manual_calibration.spin()
    if mode == 2:
        manual_calibration.on_subscriber()
        manual_calibration.spin()
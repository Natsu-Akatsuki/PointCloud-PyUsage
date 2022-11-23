import mgrs
import pymap3d as pm
import rosbag
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import NavSatFix
from tf2_msgs.msg import TFMessage
from tqdm import tqdm


def set_transform(header, frame_id, child_frame_id, x, y, z, q):
    gnss_transform = TransformStamped()
    gnss_transform.header.stamp = header.stamp
    gnss_transform.header.frame_id = frame_id
    gnss_transform.child_frame_id = child_frame_id
    gnss_transform.transform.translation.x = x
    gnss_transform.transform.translation.y = y
    gnss_transform.transform.translation.z = z
    gnss_transform.transform.rotation.x = q[0]
    gnss_transform.transform.rotation.y = q[1]
    gnss_transform.transform.rotation.z = q[2]
    gnss_transform.transform.rotation.w = q[3]
    return gnss_transform


class Converter:
    def __init__(self):
        self.input_bag = "input.bag"
        self.output_bag = "output.bag"
        self.in_gnss_topic = "/sensorgps"
        self.out_gnss_topic = "/navsat/fix"
        self.out_gnss_odom_topic = "/gnss_pose"
        self.in_lidar_topic = "/sensing/lidar/top/rslidar_points"
        self.out_lidar_topic = "/rslidar_points"

        self.m_converter = mgrs.MGRS()
        self.gnss_init = False
        self.use_local_cartesian = True

    def write(self):
        output_bag = rosbag.Bag(self.output_bag, 'w')
        input_bag = rosbag.Bag(self.input_bag, 'r')

        for topic, msg, t in tqdm(input_bag.read_messages(), total=input_bag.get_message_count()):
            if topic == self.in_gnss_topic:
                navsatfix = NavSatFix()
                navsatfix.header = msg.header
                navsatfix.header.frame_id = "gnss"
                navsatfix.latitude = msg.lat
                navsatfix.longitude = msg.lon
                navsatfix.altitude = msg.alt
                navsatfix.status.status = msg.satenum

                if not self.gnss_init:
                    self.gnss_init = True
                    if self.use_local_cartesian:
                        self.init_lat = msg.lat
                        self.init_lon = msg.lon
                        self.init_alt = msg.alt
                    else:
                        precision = 5
                        mgrs_code = self.m_converter.toMGRS(msg.lat, msg.lon, MGRSPrecision=precision)
                        self.init_local_x = int(mgrs_code[5:5 + precision])
                        self.init_local_y = int(mgrs_code[5 + precision:])

                output_bag.write(self.out_gnss_topic, navsatfix, t)

                gnss_odom = Odometry()
                gnss_odom.header.stamp = msg.header.stamp
                gnss_odom.header.frame_id = "map"  # equal to enu
                gnss_odom.child_frame_id = "gnss"

                # WGS84->ENU
                if self.use_local_cartesian:
                    local_x, local_y, local_z = pm.geodetic2enu(msg.lat, msg.lon, msg.alt, self.init_lat, self.init_lon,
                                                                self.init_alt)
                # WGS84->MGRS
                else:
                    precision = 5
                    mgrs_code = self.m_converter.toMGRS(msg.lat, msg.lon, MGRSPrecision=precision)
                    local_x = int(mgrs_code[5:5 + precision])
                    local_y = int(mgrs_code[5 + precision:])
                    local_z = 0

                gnss_odom.pose.pose.position.x = local_x
                gnss_odom.pose.pose.position.y = local_y
                gnss_odom.pose.pose.position.z = local_z

                # @ref: https://github.com/HRex39/Newton-M2-Ros-Driver/blob/main/starneto/src/starneto_mems.cpp#L162
                # 地理系下，航向角和俯仰角都是顺时针为正（以满足相关的物理意义，此处进行取反回归右手系逆时针为正的定义）
                msg.heading = 360 - msg.heading
                msg.pitch = -msg.pitch
                msg.roll = msg.roll
                # enu -> front(x)-left(y)-up(z)
                # note: 理论上下面的roll不用取反（由于原来的gps驱动取反了，此处进行补偿）
                # 步骤一：东北天->IMU右前上->IMU前左上
                map_to_imu = Rotation.from_euler('ZXY', [msg.heading, msg.pitch, -msg.roll], degrees=True)
                imu_to_ros_imu = Rotation.from_euler('ZYX', [90, 0, 0], degrees=True)
                enu_to_ros_imu = map_to_imu.as_matrix() @ imu_to_ros_imu.as_matrix()
                q = Rotation.from_matrix(enu_to_ros_imu).as_quat()

                gnss_odom.pose.pose.orientation.x = q[0]
                gnss_odom.pose.pose.orientation.y = q[1]
                gnss_odom.pose.pose.orientation.z = q[2]
                gnss_odom.pose.pose.orientation.w = q[3]

                output_bag.write(self.out_gnss_odom_topic, gnss_odom, msg.header.stamp)

                tf_msg = TFMessage()
                # enu->前左上IMU的TF
                neu_to_imu = set_transform(msg.header, "map", "imu", local_x, local_y, local_z, q)
                q_imu_to_lidar = Rotation.from_euler('ZYX', [0, 0, 0], degrees=True).as_quat()
                # 前左上IMU->前左上激光雷达的TF
                imu_to_lidar = set_transform(msg.header, "imu", "rslidar", 0, 0, 0, q_imu_to_lidar)
                tf_msg.transforms.append(neu_to_imu)
                tf_msg.transforms.append(imu_to_lidar)

                output_bag.write("/tf", tf_msg, msg.header.stamp)

            if topic == self.in_lidar_topic:
                output_bag.write(self.out_lidar_topic, msg, msg.header.stamp)

        input_bag.close()
        output_bag.close()


if __name__ == '__main__':
    converter = Converter()
    converter.write()

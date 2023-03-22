import numpy as np
import rosbag2_py
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from common_utils import get_rosbag_options, create_topic
from ampcl.ros import np_to_pointcloud2
from rclpy.serialization import deserialize_message, serialize_message
from builtin_interfaces.msg import Time


class Converter():
    def __init__(self, input_bag_path, output_bag_path, storage_id="sqlite3", serialization_format='cdr'):

        storage_options, converter_options = get_rosbag_options(input_bag_path, storage_id)

        # Seek No Filter
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # 构建主题名和消息类型的映射，此处需要结合实际情况进行修订
        # topic_types = reader.get_all_topics_and_types()
        # type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        # type_map['/livox/lidar'] = CustomMsg
        # type_map['/rslidar_points'] = PointCloud2

        storage_options, converter_options = get_rosbag_options(output_bag_path, storage_id)
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)

        # 获取读取数据的总数
        # todo：该API暂时无法使用，需要等待rosbag2_py的apt更新（不过可是使用源码安装）
        # num_msgs = reader.get_metadata().message_count

        create_topic(writer, '/livox/lidar', 'sensor_msgs/msg/PointCloud2')
        create_topic(writer, '/rslidar_points', 'sensor_msgs/msg/PointCloud2')

        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == '/livox/lidar':
                custom_msg = deserialize_message(data, CustomMsg)
                point_num = custom_msg.point_num
                pc_np = np.zeros((point_num, 4))
                for i in range(point_num):
                    pc_np[i, 0] = custom_msg.points[i].x
                    pc_np[i, 1] = custom_msg.points[i].y
                    pc_np[i, 2] = custom_msg.points[i].z
                    pc_np[i, 3] = custom_msg.points[i].reflectivity

                header = Header()
                header.frame_id = custom_msg.header.frame_id
                if 0:
                    header.stamp = custom_msg.header.stamp
                else:
                    # 使用rosbag采集时间
                    timebase_ns = t  # custom_msg.timebase
                    # 将纳秒数转换为秒数和纳秒数两个部分
                    sec = timebase_ns // 1000000000
                    nsec = timebase_ns % 1000000000
                    time_obj = Time(sec=sec, nanosec=nsec)
                    header.stamp = time_obj

                pc_ros = np_to_pointcloud2(pc_np, header, field="xyzi")

                output = serialize_message(pc_ros)
            else:
                output = data

            writer.write(topic, output, t)

        del writer


if __name__ == "__main__":
    input_bag_path = "rosbag/input"
    output_bag_path = "rosbag/output"
    converter = Converter(input_bag_path, output_bag_path)

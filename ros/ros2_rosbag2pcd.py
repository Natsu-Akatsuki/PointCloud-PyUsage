import numpy as np
import rosbag2_py
from ampcl.io import save_pointcloud
from ampcl.ros import np_to_pointcloud2
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from ampcl.ros import pointcloud2_to_np
from common_utils import get_rosbag_options
import argparse


class Converter():
    def __init__(self, input_bag_path, output_dir, storage_id="sqlite3", serialization_format='cdr'):

        storage_options, converter_options = get_rosbag_options(input_bag_path, storage_id)

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # 构建主题名和消息类型的映射，此处需要结合实际情况进行修订
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        type_map['/livox/lidar'] = PointCloud2

        # 获取读取数据的总数
        # todo：该API暂时无法使用，需要等待rosbag2_py的apt更新（现有humble版本为0.15.4 2023/4/3）
        # num_msgs = reader.get_metadata().message_count
        sequence_id = 0
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == '/livox/lidar':
                pc2_msg = deserialize_message(data, PointCloud2)
                pc_np = pointcloud2_to_np(pc2_msg)
                output_file = output_dir + f"/{str(sequence_id).zfill(6)}" + ".pcd"
                save_pointcloud(pc_np, output_file, fields="xyzi")
                sequence_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PointCloud2 messages in a ROS2 bag file to PCD files.')
    parser.add_argument('input_bag_path', type=str, help='Path to the input ROS2 bag file.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')
    args = parser.parse_args()

    converter = Converter(args.input_bag_path, args.output_dir)

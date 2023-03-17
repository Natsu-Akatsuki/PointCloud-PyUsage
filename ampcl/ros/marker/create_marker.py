import colorsys
import math
import pickle
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict
from scipy.spatial.transform import Rotation

# isort: off
try:
    import rospy

    __ROS__VERSION__ = 1
except:

    from rclpy.duration import Duration

    __ROS__VERSION__ = 2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# isort: on

class Dimensions:
    def __init__(self, box):
        self.x = float(box[3])
        self.y = float(box[4])
        self.z = float(box[5])


def create_bounding_box_model(box, stamp,
                              identity, ns="model",
                              frame_id="lidar"):
    """

    :param box: [x, y, z, l, w, h, yaw]
    :param stamp:
    :param identity:
    :param ns:
    :param frame_id:
    :return:
    """

    box_marker = Marker()
    box_marker.header.stamp = stamp
    box_marker.header.frame_id = frame_id
    box_marker.ns = ns
    box_marker.id = identity

    car_model_path = str((Path(__file__).parent / "data/model/car.dae").resolve())
    box_marker.mesh_resource = f"file://{car_model_path}"
    box_marker.mesh_use_embedded_materials = True  # Need this to use textures for mesh
    box_marker.type = box_marker.MESH_RESOURCE
    box_marker.header.frame_id = frame_id
    box_marker.action = Marker.MODIFY

    box_marker.color.r = 1.0
    box_marker.color.g = 1.0
    box_marker.color.b = 1.0
    box_marker.color.a = 0.5

    box_marker.pose.position.x = float(box[0])
    box_marker.pose.position.y = float(box[1])
    box_marker.pose.position.z = float(box[2])

    rotation = Rotation.from_euler("ZYX", [float(box[6]), 0, 0])
    quat = rotation.as_quat()
    box_marker.pose.orientation.x = quat[0]
    box_marker.pose.orientation.y = quat[1]
    box_marker.pose.orientation.z = quat[2]
    box_marker.pose.orientation.w = quat[3]

    box_marker.scale.x = 0.5
    box_marker.scale.y = 0.5
    box_marker.scale.z = 0.5

    return box_marker


def create_bounding_box_marker(box, stamp, identity, color, frame_id="lidar",
                               box_ns="shape",
                               text_ns="confident", confident=None, ground_coeff=None):
    """

    :param box: [x, y, z, l, w, h, yaw]
    :param stamp:
    :param identity:
    :param color:
    :param frame_id:
    :param box_ns:
    :param text_ns:
    :param confident:
    :param ground_coeff: 追加地面高度的补偿
    :return:
    """
    box_marker = Marker()
    box_marker.header.stamp = stamp
    box_marker.header.frame_id = frame_id
    box_marker.ns = box_ns
    box_marker.id = identity

    line_width = 0.05

    box_marker.type = Marker.LINE_LIST
    box_marker.action = Marker.MODIFY
    if ground_coeff is not None:
        A, B, C, D = ground_coeff
        height_offset = (-A * float(box[0]) + -B * float(box[1])) / C
    else:
        height_offset = 0.0

    box_marker.pose.position.x = float(box[0])
    box_marker.pose.position.y = float(box[1])
    box_marker.pose.position.z = float(box[2]) + height_offset

    dimensions = Dimensions(box)
    box_marker.points = calc_bounding_box_line_list(dimensions)

    rotation = Rotation.from_euler("ZYX", [float(box[6]), 0, 0])
    quat = rotation.as_quat()
    box_marker.pose.orientation.x = quat[0]
    box_marker.pose.orientation.y = quat[1]
    box_marker.pose.orientation.z = quat[2]
    box_marker.pose.orientation.w = quat[3]

    if __ROS__VERSION__ == 1:
        box_marker.lifetime = rospy.Duration(0)
    elif __ROS__VERSION__ == 2:
        box_marker.lifetime = Duration(seconds=0).to_msg()

    box_marker.scale.x = line_width
    box_marker.color.a = 0.999
    box_marker.color.r = color[0]
    box_marker.color.g = color[1]
    box_marker.color.b = color[2]

    if confident is not None:
        confident_marker = Marker()
        confident_marker.header.stamp = stamp
        confident_marker.header.frame_id = frame_id
        confident_marker.ns = text_ns
        confident_marker.id = identity
        confident_marker.action = Marker.ADD
        confident_marker.type = Marker.TEXT_VIEW_FACING

        confident_marker.color.a = 0.999
        confident_marker.color.r = 0.0
        confident_marker.color.g = 0.0
        confident_marker.color.b = 1.0
        confident_marker.scale.z = 0.8  # 设置字体大小
        confident_marker.pose.orientation.w = 1.0

        confident_marker.pose.position.x = float(box[0])
        confident_marker.pose.position.y = float(box[1])
        confident_marker.pose.position.z = float(box[2]) + dimensions.z / 2.0 + 0.5 + height_offset
        confident_marker.text = str(np.floor(confident * 100) / 100)

        if __ROS__VERSION__ == 1:
            confident_marker.lifetime = rospy.Duration(0)
        elif __ROS__VERSION__ == 2:
            confident_marker.lifetime = Duration(seconds=0).to_msg()
        return box_marker, confident_marker
    else:
        return box_marker, None


def calc_bounding_box_line_list(dimensions):
    points = []
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    # up surface
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = dimensions.z / 2.0
    points.append(point)

    # down surface
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    point = Point()
    point.x = dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)
    point = Point()
    point.x = -dimensions.x / 2.0
    point.y = -dimensions.y / 2.0
    point.z = -dimensions.z / 2.0
    points.append(point)

    return points

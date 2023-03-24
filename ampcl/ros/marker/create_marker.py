# isort: off
try:
    import rospy

    __ROS__VERSION__ = 1
except:

    from rclpy.duration import Duration

    __ROS__VERSION__ = 2

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
# isort: on
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def set_lifetime(marker, seconds=0.0):
    if __ROS__VERSION__ == 1:
        marker.lifetime = rospy.Duration(seconds)
    elif __ROS__VERSION__ == 2:
        marker.lifetime = Duration(seconds=seconds).to_msg()


class Dimensions:
    def __init__(self, box):
        self.x = float(box[3])
        self.y = float(box[4])
        self.z = float(box[5])


def create_box3d_model(box3d, stamp,
                       identity, ns="model",
                       frame_id="lidar"):
    """

    :param box3d: [x, y, z, l, w, h, yaw]
    :param stamp:
    :param identity:
    :param ns:
    :param frame_id:
    :return:
    """

    box3d_marker = Marker()
    box3d_marker.header.stamp = stamp
    box3d_marker.header.frame_id = frame_id
    box3d_marker.ns = ns
    box3d_marker.id = identity

    car_model_path = str((Path(__file__).parent / "data/model/car.dae").resolve())
    box3d_marker.mesh_resource = f"file://{car_model_path}"
    box3d_marker.mesh_use_embedded_materials = True  # Need this to use textures for mesh
    box3d_marker.type = box3d_marker.MESH_RESOURCE
    box3d_marker.header.frame_id = frame_id
    box3d_marker.action = Marker.MODIFY
    box3d_marker.scale.x = 0.5
    box3d_marker.scale.y = 0.5
    box3d_marker.scale.z = 0.5

    box3d = box3d.astype(np.float32)
    rotation = Rotation.from_euler("ZYX", [box3d[6], 0, 0])
    quat = rotation.as_quat()
    set_color(box3d_marker, (1.0, 1.0, 1.0, 0.5))
    set_position(box3d_marker, box3d[0:3])
    set_orientation(box3d_marker, quat)

    return box3d_marker


def create_box3d_marker(box3d, stamp, frame_id="lidar",
                        box3d_ns="shape", box3d_color=(0.0, 0.0, 0.0), box3d_id=0,
                        class_ns="class", class_name=None,
                        tracker_ns="tracker", tracker_id=None,
                        confidence_ns="confidence", confidence=None,
                        plane_model=None,
                        box3d_wise_height_offset=0.0):
    """
    :param box3d:  [x, y, z, l, w, h, yaw] 激光雷达系
    :param stamp:
    :param identity:
    :param color:
    :param frame_id:
    :param box3d_ns:
    :param text_ns:
    :param confidence:
    :param plane_model: 追加地面高度的补偿
    :return:
    """

    box3d_marker = Marker()
    box3d_marker.header.stamp = stamp
    box3d_marker.header.frame_id = frame_id
    box3d_marker.ns = box3d_ns
    box3d_marker.id = box3d_id
    box3d_marker.type = Marker.LINE_LIST
    box3d_marker.action = Marker.MODIFY

    line_width = 0.05
    box3d_marker.scale.x = line_width

    box3d = box3d.astype(np.float32)
    if plane_model is not None:
        A, B, C, D = plane_model
        height_offset = -(A * box3d[0] + B * box3d[1] + D) / C - box3d_wise_height_offset
    else:
        height_offset = -box3d_wise_height_offset
    box3d_z = box3d[2] + height_offset
    set_position(box3d_marker, [box3d[0], box3d[1], box3d_z])

    dimensions = Dimensions(box3d)
    box3d_marker.points = calc_bounding_box_line_list(dimensions)

    rotation = Rotation.from_euler("ZYX", [box3d[6], 0, 0])
    quat = rotation.as_quat()

    set_orientation(box3d_marker, quat[:4])
    set_lifetime(box3d_marker)
    set_color(box3d_marker, (box3d_color[0], box3d_color[1], box3d_color[2], 0.999))

    marker_dict = {"box": box3d_marker}

    if confidence is not None:
        confident_marker = Marker()
        confident_marker.header.stamp = stamp
        confident_marker.header.frame_id = frame_id
        confident_marker.ns = confidence_ns
        confident_marker.id = box3d_id
        confident_marker.action = Marker.ADD
        confident_marker.type = Marker.TEXT_VIEW_FACING
        confident_marker.text = str(np.floor(confidence * 100) / 100)
        confident_marker.scale.z = 0.8  # 设置字体大小

        box3d_x = box3d[0]
        box3d_y = box3d[1]
        box3d_z = box3d[2] + dimensions.z / 2.0 + 0.5 + height_offset

        set_lifetime(confident_marker)
        set_color(confident_marker, (0.0, 0.0, 1.0, 0.999))
        set_orientation(confident_marker, [0.0, 0.0, 0.0, 1.0])
        set_position(confident_marker, (box3d_x, box3d_y, box3d_z))

        marker_dict["confident_marker"] = confident_marker

    if tracker_id is not None:
        tracker_marker = Marker()
        tracker_marker.header.stamp = stamp
        tracker_marker.header.frame_id = frame_id
        tracker_marker.ns = tracker_ns
        tracker_marker.id = box3d_id
        tracker_marker.action = Marker.ADD
        tracker_marker.type = Marker.TEXT_VIEW_FACING
        tracker_marker.scale.z = 0.8  # 设置字体大小

        box3d_x = box3d[0]
        box3d_y = box3d[1]
        box3d_z = box3d[2] + dimensions.z / 2.0 + 0.5 + height_offset

        set_color(tracker_marker, (0.0, 0.0, 1.0, 0.999))
        set_orientation(tracker_marker, [0.0, 0.0, 0.0, 1.0])
        set_position(tracker_marker, (box3d_x, box3d_y, box3d_z))
        tracker_marker.text = f"ID {tracker_id}"
        set_lifetime(tracker_marker, seconds=0.0)
        marker_dict["tracker_marker"] = tracker_marker

    return marker_dict


def calc_bounding_box_line_list(dimensions):
    points = []
    half_x = dimensions.x / 2.0
    half_y = dimensions.y / 2.0
    half_z = dimensions.z / 2.0

    # add all corner points
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                point = Point()
                point.x = x * half_x
                point.y = y * half_y
                point.z = z * half_z
                points.append(point)

    marker_points = []
    # add all edge points
    for i, j in [(0, 1), (0, 2), (0, 4), (1, 3),
                 (1, 5), (2, 3), (2, 6), (3, 7),
                 (4, 5), (4, 6), (5, 7), (6, 7),
                 (4, 7), (5, 6)]:
        p1 = points[i]
        p2 = points[j]
        marker_points.append(p1)
        marker_points.append(p2)

    return marker_points


def create_distance_marker(frame_id="lidar", circle_num=15, distance_delta=1.0):
    marker_array = MarkerArray()
    for circle_id in range(circle_num):
        circle = Marker()

        # meta information
        circle.header.frame_id = frame_id
        circle.ns = "distance_circle"
        circle.id = circle_id
        circle.action = Marker.ADD
        circle.type = Marker.LINE_STRIP
        set_lifetime(circle, seconds=0.0)

        # geometry information
        set_orientation(circle)
        for theta in np.arange(0, 2 * 3.24, 0.1):
            point = Point()
            point.x = circle_id * distance_delta * math.cos(theta)
            point.y = circle_id * distance_delta * math.sin(theta)
            point.z = 0.0
            circle.points.append(point)

        # color information
        set_color(circle, (0.5, 0.5, 0.5, 0.9))
        # style information
        circle.scale.x = 0.1  # line width

        marker_array.markers.append(circle)

    for text_id in range(circle_num):
        text_marker = Marker()

        # meta information
        text_marker.header.frame_id = frame_id
        text_marker.ns = "distance_text"
        text_marker.id = text_id
        text_marker.action = Marker.ADD
        text_marker.type = Marker.TEXT_VIEW_FACING
        set_lifetime(text_marker, seconds=0.0)

        # geometry information
        theta = -45 * math.pi / 180
        text_marker.pose.position.x = (text_id * distance_delta) * math.cos(theta)
        text_marker.pose.position.y = (text_id * distance_delta) * math.sin(theta)
        text_marker.pose.position.z = 0.0
        # color information
        set_color(text_marker, (0.0, 0.0, 0.0, 0.9))
        # style information
        text_marker.scale.z = 2.0  # font size
        text_marker.text = f"{text_id * distance_delta}"

        marker_array.markers.append(text_marker)

    return marker_array


def create_point(position=(0, 0, 0)):
    p = Point()
    p.x = position[0]
    p.y = position[1]
    p.z = position[2]
    return p


def set_color(marker, rgba):
    """
    Set the color of a marker
    :param marker:
    :param rgba:
    """
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]  # Required, otherwise rviz can not be displayed


def set_orientation(marker, orientation=(0.0, 0.0, 0.0, 1.0)):
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]  # suppress the uninitialized quaternion, assuming identity warning


def set_position(marker, position=(0.0, 0.0, 0.0)):
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])

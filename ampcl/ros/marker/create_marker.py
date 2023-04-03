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
import colorsys
import math
import random
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...filter import get_indices_of_points_inside


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
                        class_text_ns="text/class_id", class_text=None,
                        tracker_text_ns="text/tracker_id", tracker_text=None,
                        confidence_text_ns="text/confidence", confidence_text=None):
    box3d_marker = Marker()
    box3d_marker.header.stamp = stamp
    box3d_marker.header.frame_id = frame_id
    box3d_marker.ns = box3d_ns
    box3d_marker.id = box3d_id
    box3d_marker.type = Marker.LINE_LIST
    box3d_marker.action = Marker.MODIFY

    line_width = 0.05
    box3d_marker.scale.x = line_width

    dimensions = Dimensions(box3d)
    box3d_marker.points = calc_bounding_box_line_list(dimensions)

    rotation = Rotation.from_euler("ZYX", [box3d[6], 0, 0])
    quat = rotation.as_quat()

    set_color(box3d_marker, (box3d_color[0], box3d_color[1], box3d_color[2], 0.999))
    set_orientation(box3d_marker, quat[:4])
    set_position(box3d_marker, box3d[0:3])

    set_lifetime(box3d_marker, seconds=0.0)

    marker_dict = {"box": box3d_marker}

    if confidence_text is not None:
        confident_marker = Marker()
        confident_marker.header.stamp = stamp
        confident_marker.header.frame_id = frame_id
        confident_marker.ns = confidence_text_ns
        confident_marker.id = box3d_id
        confident_marker.action = Marker.ADD
        confident_marker.type = Marker.TEXT_VIEW_FACING

        set_color(confident_marker, (0.0, 0.0, 1.0, 0.999))
        set_orientation(confident_marker, [0.0, 0.0, 0.0, 1.0])

        box3d_x = box3d[0]
        box3d_y = box3d[1]
        box3d_z = box3d[2] + dimensions.z / 2.0 + 0.5
        set_position(confident_marker, (box3d_x, box3d_y, box3d_z))

        confident_marker.text = str(np.floor(confidence_text * 100) / 100)
        confident_marker.scale.z = 0.8  # 设置字体大小

        set_lifetime(confident_marker, seconds=0.0)
        marker_dict["confident_marker"] = confident_marker

    if tracker_text is not None:
        tracker_marker = Marker()
        tracker_marker.header.stamp = stamp
        tracker_marker.header.frame_id = frame_id
        tracker_marker.ns = tracker_text_ns
        tracker_marker.id = box3d_id
        tracker_marker.action = Marker.ADD
        tracker_marker.type = Marker.TEXT_VIEW_FACING

        set_color(tracker_marker, (0.0, 0.0, 1.0, 0.999))
        set_orientation(tracker_marker, [0.0, 0.0, 0.0, 1.0])

        box3d_x = box3d[0]
        box3d_y = box3d[1]
        box3d_z = box3d[2] + dimensions.z / 2.0 + 0.5
        set_position(tracker_marker, (box3d_x, box3d_y, box3d_z))

        tracker_marker.text = f"ID:{tracker_text}"
        tracker_marker.scale.z = 0.8  # 设置字体大小

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
        theta = -15 * math.pi / 180
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
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])  # Required, otherwise rviz can not be displayed


def set_orientation(marker, orientation=(0.0, 0.0, 0.0, 1.0)):
    marker.pose.orientation.x = orientation[0]
    marker.pose.orientation.y = orientation[1]
    marker.pose.orientation.z = orientation[2]
    marker.pose.orientation.w = orientation[3]  # suppress the uninitialized quaternion, assuming identity warning


def set_position(marker, position=(0.0, 0.0, 0.0)):
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])


def cls_id_to_color(cls_id):
    color_maps = {-1: [0.5, 0.5, 0.5],
                  0: [1.0, 0.0, 0.0],
                  1: [0.0, 1.0, 0.0],  # 绿：Vehicle
                  2: [0.0, 0.0, 1.0],  # 蓝：Pedestrian
                  3: [1.0, 0.8, 0.0],  # 土黄：Cyclist
                  4: [1.0, 0.0, 1.0]
                  }
    if cls_id not in color_maps.keys():
        return [0.5, 1.0, 0.5]

    return color_maps[cls_id]


def random_colors(N, bright=True):
    """
    @from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


instance_colors = random_colors(100)


def instance_id_to_color(instance_id):
    return instance_colors[int(instance_id) % 100]


def create_gt_detection_box3d_marker_array(pointcloud, box3d_lidar, pc_color=None,
                                           box3d_arr_marker=None, stamp=None, frame_id=None,
                                           box3d_ns="gt/detection/box3d",
                                           plane_model=None, box3d_wise_height_offset=None,
                                           use_color=True,
                                           ):
    """
    :param pointcloud: 真值框固定为红色
    :param box3d_lidar: [N,8] [x, y, z, l, w, h, yaw, cls_id]
                     or [N,9] [x, y, z, l, w, h, yaw, cls_id, tracker_id]
    :param pc_color: 修订传进去的点云颜色字段（为引用）
    :param box3d_arr_marker:
    :param stamp:
    :param frame_id:
    :param box3d_ns:
    :param tracker_text_ns:
    :param plane_model:
    :param box3d_wise_height_offset:
    :param use_color:
    :return:
    """
    if box3d_lidar.shape[0] == 0:
        return box3d_lidar

    box3d_lidar = box3d_lidar.copy()
    for i in range(box3d_lidar.shape[0]):
        a_box3d_lidar = box3d_lidar[i]

        if a_box3d_lidar[7] == -1:
            continue

        # 提取box中的激光点
        if pc_color is not None:
            box3d_color = cls_id_to_color(a_box3d_lidar[7])
            inside_points_idx = get_indices_of_points_inside(pointcloud, a_box3d_lidar, margin=0.1)
            pc_color[inside_points_idx] = box3d_color

        # 对marker的高度进行的修正
        if plane_model is not None and box3d_wise_height_offset is not None:
            A, B, C, D = plane_model
            a_box3d_lidar[2] += -(A * a_box3d_lidar[0] + B * a_box3d_lidar[1] + D) / C - box3d_wise_height_offset
        elif box3d_wise_height_offset is not None:
            a_box3d_lidar[2] += -box3d_wise_height_offset
        box3d_color = (1.0, 0.0, 0.0)
        marker_dict = create_box3d_marker(a_box3d_lidar,
                                          stamp, frame_id=frame_id,
                                          box3d_id=i, box3d_ns=box3d_ns, box3d_color=box3d_color)

        box3d_arr_marker.markers += list(marker_dict.values())
    return box3d_lidar


def create_pred_detection_box3d_marker_array(pointcloud, box3d_lidar, pc_color=None, score=None,
                                             box3d_arr_marker=None, stamp=None, frame_id=None,
                                             box3d_ns="pred/detection/box3d",
                                             plane_model=None, box3d_wise_height_offset=None,
                                             ):
    """ 预测预测框的颜色与类别颜色所绑定
    :param pointcloud:
    :param box3d_lidar: [N,8] [x, y, z, l, w, h, yaw, cls_id]
                     or [N,9] [x, y, z, l, w, h, yaw, cls_id, tracker_id]
    :param pc_color: 修订传进去的点云颜色字段（为引用）
    :param box3d_arr_marker:
    :param stamp:
    :param frame_id:
    :param box3d_ns:
    :param tracker_text_ns:
    :param plane_model:
    :param box3d_wise_height_offset:
    :param use_color:
    :return:
    """
    if box3d_lidar.shape[0] == 0:
        return box3d_lidar

    box3d_lidar = box3d_lidar.copy()
    for i in range(box3d_lidar.shape[0]):
        a_box3d_lidar = box3d_lidar[i]

        box3d_color = cls_id_to_color(a_box3d_lidar[7])

        # 提取box中的激光点
        if pc_color is not None:
            inside_points_idx = get_indices_of_points_inside(pointcloud, a_box3d_lidar, margin=0.1)
            pc_color[inside_points_idx] = box3d_color

        # 对marker的高度进行的修正
        if plane_model is not None and box3d_wise_height_offset is not None:
            A, B, C, D = plane_model
            a_box3d_lidar[2] += -(A * a_box3d_lidar[0] + B * a_box3d_lidar[1] + D) / C - box3d_wise_height_offset
        elif box3d_wise_height_offset is not None:
            a_box3d_lidar[2] += -box3d_wise_height_offset

        marker_dict = create_box3d_marker(a_box3d_lidar,
                                          stamp, frame_id=frame_id,
                                          box3d_id=i, box3d_ns=box3d_ns, box3d_color=box3d_color,
                                          confidence_text_ns="pred/detection/text/confidence",
                                          confidence_text=score[i])

        box3d_arr_marker.markers += list(marker_dict.values())
    return box3d_lidar


def create_pred_tracker_box3d_marker_array(pointcloud, box3d_lidar, pc_color=None,
                                           box3d_arr_marker=None, stamp=None, frame_id=None,
                                           box3d_ns="pred/tracker/box3d",
                                           tracker_text_ns="pred/tracker/text/tracker_id",
                                           plane_model=None, box3d_wise_height_offset=None,
                                           ):
    """ 预测跟踪框默认为红色（其他颜色需修改对应源码）
    :param pointcloud:
    :param box3d_lidar: [N,8] [x, y, z, l, w, h, yaw, cls_id]
                     or [N,9] [x, y, z, l, w, h, yaw, cls_id, tracker_id]
    :param pc_color: 修订传进去的点云颜色字段（为引用）
    :param box3d_arr_marker:
    :param stamp:
    :param frame_id:
    :param box3d_ns:
    :param tracker_text_ns:
    :param plane_model:
    :param box3d_wise_height_offset:
    :param use_color:
    :return:
    """
    if box3d_lidar.shape[0] == 0:
        return box3d_lidar

    box3d_lidar = box3d_lidar.copy()
    for i in range(box3d_lidar.shape[0]):
        a_box3d_lidar = box3d_lidar[i]

        box3d_color = instance_id_to_color(a_box3d_lidar[8])
        tracker_text = a_box3d_lidar[8]

        # 提取box中的激光点
        if pc_color is not None:
            inside_points_idx = get_indices_of_points_inside(pointcloud, a_box3d_lidar, margin=0.1)
            pc_color[inside_points_idx] = box3d_color

        # 对Marker的高度进行的修正
        if plane_model is not None and box3d_wise_height_offset is not None:
            A, B, C, D = plane_model
            a_box3d_lidar[2] += -(A * a_box3d_lidar[0] + B * a_box3d_lidar[1] + D) / C - box3d_wise_height_offset
        elif box3d_wise_height_offset is not None:
            a_box3d_lidar[2] += -box3d_wise_height_offset

        box3d_color = (1.0, 0.0, 0.0)
        marker_dict = create_box3d_marker(a_box3d_lidar,
                                          stamp, frame_id=frame_id,
                                          box3d_id=i, box3d_ns=box3d_ns, box3d_color=box3d_color,
                                          tracker_text_ns=tracker_text_ns, tracker_text=tracker_text)

        box3d_arr_marker.markers += list(marker_dict.values())
    return box3d_lidar

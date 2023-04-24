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
                        text_ns="text", text=None, text_size=0.8,
                        line_width=0.05):
    box3d_marker = Marker()
    box3d_marker.header.stamp = stamp
    box3d_marker.header.frame_id = frame_id
    box3d_marker.ns = box3d_ns
    box3d_marker.id = box3d_id
    box3d_marker.type = Marker.LINE_LIST
    box3d_marker.action = Marker.MODIFY

    box3d_marker.scale.x = line_width

    dimensions = Dimensions(box3d)
    box3d_marker.points = calc_bounding_box_line_list(dimensions)

    rotation = Rotation.from_euler("ZYX", [box3d[6], 0, 0])
    quat = rotation.as_quat()

    set_color(box3d_marker, (box3d_color[0], box3d_color[1], box3d_color[2], 0.999))
    set_orientation(box3d_marker, quat[:4])
    set_position(box3d_marker, box3d[0:3])

    set_lifetime(box3d_marker, seconds=0.0)

    marker_dict = {"box3d_marker": box3d_marker}

    if text_ns is not None and text is not None:
        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = frame_id
        text_marker.ns = text_ns
        text_marker.id = box3d_id
        text_marker.action = Marker.ADD
        text_marker.type = Marker.TEXT_VIEW_FACING

        set_color(text_marker, (0.0, 0.0, 1.0, 0.999))
        set_orientation(text_marker, [0.0, 0.0, 0.0, 1.0])

        box3d_x = box3d[0]
        box3d_y = box3d[1]
        box3d_z = box3d[2] + dimensions.z / 2.0 + 0.5
        set_position(text_marker, (box3d_x, box3d_y, box3d_z))

        text_marker.text = text
        text_marker.scale.z = text_size  # 设置字体大小

        set_lifetime(text_marker, seconds=0.0)
        marker_dict["text_marker"] = text_marker

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


def kitti_cls_id_to_color(cls_id):
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
    random.seed(233)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


instance_colors = random_colors(100)


def instance_id_to_color(instance_id):
    return instance_colors[int(instance_id) % 100]


def create_box3d_marker_array(box3d_marker_array, box3d_lidar,
                              stamp, frame_id,
                              box3d_ns="box3d",
                              color_method=None,
                              line_width=0.05,
                              text_ns="text", text_list=None, text_size=0.8,
                              pc_np=None, pc_color=None):
    if box3d_lidar.shape[0] == 0:
        return
    if len(box3d_lidar.shape) != 2:
        raise ValueError("box3d_lidar must be a 2D array")

    box3d_color = [0, 0, 0]
    for i in range(box3d_lidar.shape[0]):
        a_box3d_lidar = box3d_lidar[i]
        text = str(text_list[i]) if text_list is not None else None

        if color_method == "class":
            cls_id = int(a_box3d_lidar[7])
            box3d_color = kitti_cls_id_to_color(cls_id)
        elif color_method == "instance":
            instance_id = int(a_box3d_lidar[8])
            box3d_color = instance_id_to_color(instance_id)
        elif color_method is None:
            box3d_color = box3d_color
        else:
            pass

        if pc_np is not None and pc_color is not None:
            inside_points_idx = get_indices_of_points_inside(pc_np, a_box3d_lidar, margin=0.1)
            pc_color[inside_points_idx] = box3d_color

        marker_dict = create_box3d_marker(a_box3d_lidar,
                                          stamp, frame_id=frame_id,
                                          box3d_id=i, box3d_ns=box3d_ns,
                                          box3d_color=box3d_color,
                                          text_ns=text_ns, text=text,
                                          line_width=line_width,
                                          text_size=text_size)

        box3d_marker_array.markers += list(marker_dict.values())


def init_marker_array():
    box3d_marker_array = MarkerArray()
    empty_marker = Marker()
    empty_marker.action = Marker.DELETEALL
    box3d_marker_array.markers.append(empty_marker)
    return box3d_marker_array


def create_convex_hull_marker(convex_hull, stamp, frame_id="lidar",
                              convex_hull_ns="shape",
                              convex_hull_color=(0.0, 0.0, 0.0), convex_hull_id=0,
                              text_ns="text", text=None, text_size=0.8,
                              line_width=0.05):
    convex_hull_marker = Marker()
    convex_hull_marker.header.stamp = stamp
    convex_hull_marker.header.frame_id = frame_id
    convex_hull_marker.ns = convex_hull_ns
    convex_hull_marker.id = convex_hull_id
    convex_hull_marker.type = Marker.LINE_LIST
    convex_hull_marker.action = Marker.MODIFY

    convex_hull_marker.scale.x = line_width

    z_upper_boundary = np.zeros((convex_hull.footprint.shape[0]), dtype=np.float32) \
                       + convex_hull.dimension_h / 2 + convex_hull.centroid[2]
    z_lower_boundary = np.zeros((convex_hull.footprint.shape[0]), dtype=np.float32) - convex_hull.dimension_h / 2 \
                       + convex_hull.centroid[2]

    z_upper_boundary = np.hstack((convex_hull.footprint, z_upper_boundary[:, np.newaxis]))
    z_lower_boundary = np.hstack((convex_hull.footprint, z_lower_boundary[:, np.newaxis]))

    points = np.vstack((z_upper_boundary, z_lower_boundary))

    # build line
    lines = []
    for i in range(convex_hull.footprint.shape[0] - 1):
        lines.append([i, (i + 1)])
    lines.append([convex_hull.footprint.shape[0] - 1, 0])

    for i in range(convex_hull.footprint.shape[0], points.shape[0] - 1):
        lines.append([i, (i + 1)])
    lines.append([points.shape[0] - 1, convex_hull.footprint.shape[0]])

    for i in range(convex_hull.footprint.shape[0] - 1):
        lines.append([i, (i + convex_hull.footprint.shape[0])])

    for i in range(len(lines)):
        point = Point()
        point.x = float(points[lines[i][0], 0])
        point.y = float(points[lines[i][0], 1])
        point.z = float(points[lines[i][0], 2])
        convex_hull_marker.points.append(point)

        point = Point()
        point.x = float(points[lines[i][1], 0])
        point.y = float(points[lines[i][1], 1])
        point.z = float(points[lines[i][1], 2])
        convex_hull_marker.points.append(point)

    set_color(convex_hull_marker, (convex_hull_color[0], convex_hull_color[1], convex_hull_color[2], 0.999))
    set_lifetime(convex_hull_marker, seconds=0.0)

    marker_dict = {"convex_hull_marker": convex_hull_marker}

    text_marker = Marker()
    text_marker.header.stamp = stamp
    text_marker.header.frame_id = frame_id
    text_marker.ns = text_ns
    text_marker.id = convex_hull_id
    text_marker.action = Marker.ADD
    text_marker.type = Marker.TEXT_VIEW_FACING

    set_color(text_marker, (0.0, 0.0, 1.0, 0.999))
    set_orientation(text_marker, [0.0, 0.0, 0.0, 1.0])

    pos_x = convex_hull.centroid[0]
    pos_y = convex_hull.centroid[1]
    pos_z = convex_hull.centroid[2] + convex_hull.dimension_h / 2.0 + 0.5
    set_position(text_marker, (pos_x, pos_y, pos_z))

    distance = np.linalg.norm(convex_hull.centroid[:3])
    text_marker.text = f"{distance:.1f}"
    text_marker.scale.z = text_size  # 设置字体大小

    set_lifetime(text_marker, seconds=0.0)
    marker_dict["text_marker"] = text_marker

    return marker_dict


def create_convex_hull_marker_array(box3d_marker_array, convex_hull_list,
                                    stamp, frame_id,
                                    convex_hull_ns="box3d",
                                    color_method=None,
                                    line_width=0.05,
                                    text_ns="text", text_list=None, text_size=0.8,
                                    pc_np=None, pc_color=None):
    if len(convex_hull_list) == 0:
        return

    convex_hull_color = [1, 0, 0]
    for i in range(len(convex_hull_list)):
        convex_hull = convex_hull_list[i]
        text = str(text_list[i]) if text_list is not None else None

        marker_dict = create_convex_hull_marker(convex_hull,
                                                stamp, frame_id=frame_id,
                                                convex_hull_id=i, convex_hull_ns=convex_hull_ns,
                                                convex_hull_color=convex_hull_color,
                                                text_ns=text_ns, text=text,
                                                line_width=line_width,
                                                text_size=text_size)

        box3d_marker_array.markers += list(marker_dict.values())

import yaml
from easydict import EasyDict
from geometry_msgs.msg import Point

from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from pathlib import Path
import pickle

try:
    import rospy

    __ROS__VERSION__ = 1
except:

    from rclpy.duration import Duration

    __ROS__VERSION__ = 2


class Config:
    def __init__(self, kitti_cfg_file):
        cfg_dict = {}
        with open(kitti_cfg_file, 'r') as f:
            kitti_cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg_dict.update(kitti_cfg)
        cfg_dict = EasyDict(cfg_dict)

        # DatasetParam
        self.dataset_dir = cfg_dict.DatasetParam.dataset_dir
        self.gt_file = cfg_dict.DatasetParam.gt_file
        self.predict_file = cfg_dict.DatasetParam.predict_file

        # ROSParam
        self.pointcloud_topic = cfg_dict.ROSParam.pointcloud_topic
        self.image_topic = cfg_dict.ROSParam.image_topic
        self.bounding_box_topic = cfg_dict.ROSParam.bounding_box_topic
        self.frame = cfg_dict.ROSParam.frame

        # VisParam
        self.auto_update = cfg_dict.VisParam.auto
        self.update_time = cfg_dict.VisParam.update_time


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


def id_to_color(cls_id):
    color_maps = {1: [0.5, 0.5, 0.5],
                  2: [1.0, 0.0, 0.0],
                  3: [0.0, 1.0, 0.0],
                  4: [0.0, 0.0, 1.0],
                  5: [1.0, 1.0, 0.0],
                  6: [1.0, 0.0, 1.0]
                  }
    if cls_id not in color_maps.keys():
        return [0.5, 1.0, 0.5]

    return color_maps[cls_id]


def open_pkl_file(infos_path):
    with open(str(infos_path), 'rb') as f:
        infos = pickle.load(f)
    return infos


class Dimensions:
    def __init__(self, box):
        self.x = float(box[3])
        self.y = float(box[4])
        self.z = float(box[5])


def create_bounding_box_model(box, stamp, identity, frame_id="lidar"):
    box_marker = Marker()
    box_marker.header.stamp = stamp
    box_marker.header.frame_id = frame_id
    box_marker.ns = "model"
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


def clear_bounding_box_marker(stamp, identity, ns="uname", frame_id="lidar"):
    box_marker = Marker()
    box_marker.header.stamp = stamp
    box_marker.header.frame_id = frame_id

    box_marker.ns = ns
    box_marker.id = identity

    box_marker.action = Marker.DELETEALL
    if __ROS__VERSION__ == 1:
        box_marker.lifetime = rospy.Duration(0.02)
    elif __ROS__VERSION__ == 2:
        box_marker.lifetime = Duration(seconds=0.02).to_msg()

    return box_marker


def create_bounding_box_marker(box, stamp, identity, color, frame_id="lidar"):
    box_marker = Marker()
    box_marker.header.stamp = stamp
    box_marker.header.frame_id = frame_id
    box_marker.ns = "shape"
    box_marker.id = identity

    line_width = 0.05

    box_marker.type = Marker.LINE_LIST
    box_marker.action = Marker.MODIFY

    box_marker.pose.position.x = float(box[0])
    box_marker.pose.position.y = float(box[1])
    box_marker.pose.position.z = float(box[2])

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

    return box_marker


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

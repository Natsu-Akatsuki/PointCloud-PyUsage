import numpy as np
from scipy.spatial.transform import Rotation


def get_indices_of_points_inside(pointcloud, box, margin=0.0):
    """ Find indices of points inside the bbox

    :param pointcloud: 激光雷达系的激光点
    :param margin: margin for the bbox to include boundary points, defaults to 0.0
    :return: indices of input points that are inside the bbox
    """
    # Axis align points and bbox boundary for easier filtering
    # This is 4x faster than `points = np.dot(points, self.rotation_matrix)`
    pointcloud_xyz = pointcloud[:, :3]
    rotation_mat = Rotation.from_euler("ZYX", [float(box[6]), 0, 0]).as_matrix()

    cx = float(box[0])
    cy = float(box[1])
    cz = float(box[2])
    t_vec = np.array([cx, cy, cz])
    l = float(box[3])
    w = float(box[4])
    h = float(box[5])

    """ corners_3d format. Facing forward: (0-4-7-3) = forward
      0 -------- 3
     /|         /|
    1 -------- 2 .
    | |        | |
    . 4 -------- 7
    |/         |/
    5 -------- 6
    """

    corners_3d_no0 = np.array([l, w, h]) / 2
    corners_3d_no6 = np.array([-l, -w, -h]) / 2

    t_offset = -rotation_mat.T @ t_vec
    # This is 4x faster than points = points @ self.rotation_matrix, amazing...
    points = (rotation_mat.T @ pointcloud_xyz.T).T + t_offset

    mask_coordinates_inside = np.logical_and(
        points <= corners_3d_no0 + margin, points >= corners_3d_no6 - margin)
    return np.flatnonzero(np.all(mask_coordinates_inside, axis=1))

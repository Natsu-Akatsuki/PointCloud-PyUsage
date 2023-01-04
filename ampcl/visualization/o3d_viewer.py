import argparse

import open3d as o3d
from colorama import Fore
from ..io import load_pointcloud
from .intensity_to_color import intensity_to_color_o3d
import numpy as np


def o3d_viewer_from_pointcloud(pointcloud, is_normalized=False, colors=None):
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])

    if pointcloud.shape[1] == 4:
        if colors is None:
            # 没有显式提供颜色时，则使用强度伪彩色
            intensity = pointcloud[:, 3].astype(np.int32)
            colors = intensity_to_color_o3d(intensity, is_normalized=is_normalized)
        pointcloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud_o3d])


def o3d_viewer_from_file(file_path, is_normalized=False):
    pointcloud_np = load_pointcloud(file_path)
    print(Fore.GREEN + f"[IO] pointcloud shape is {pointcloud_np.shape}")
    o3d_viewer_from_pointcloud(pointcloud_np, is_normalized)


def visualize_point_with_sphere(points, radius=0.005, resolution=10, color=(1, 0, 0)):
    """
    用球体的方式可视化点云（可控制激光点大小）
    :param points:
    :return:
    """
    o3d_object = []
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        sphere.translate(point[:3])
        sphere.paint_uniform_color(color)
        o3d_object.append(sphere)
    return o3d_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", action="store", help="pointcloud file name")
    parser.add_argument("-n", "--normalized", action='store_true', default=False, help="use normalized intensity")
    args = parser.parse_args()

    o3d_viewer_from_file(args.name, args.normalized)


if __name__ == '__main__':
    main()

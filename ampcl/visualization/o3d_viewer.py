import argparse

import open3d as o3d
from colorama import Fore
from ..io import load_pointcloud
from .intensity_to_color import intensity_to_color_o3d


def o3d_viewer_from_pointcloud(pointcloud, is_normalized=False, colors=None):
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])

    if pointcloud.shape[1] == 4:
        if not colors:
            color_o3d = intensity_to_color_o3d(pointcloud[:, 3], is_normalized=is_normalized)
        pointcloud_o3d.colors = o3d.utility.Vector3dVector(color_o3d)
    o3d.visualization.draw_geometries([pointcloud_o3d])


def o3d_viewer_from_file(file_path, is_normalized=False):
    pointcloud_np = load_pointcloud(file_path)
    print(Fore.GREEN + f"[IO] pointcloud shape is {pointcloud_np.shape}")
    o3d_viewer_from_pointcloud(pointcloud_np, is_normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", action="store", help="pointcloud file name")
    parser.add_argument("-n", "--normalized", action='store_true', default=False, help="use normalized intensity")
    args = parser.parse_args()

    o3d_viewer_from_file(args.name, args.normalized)


if __name__ == '__main__':
    main()

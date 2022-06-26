import numpy as np
from matplotlib import pyplot as plt
from numba import vectorize, float32, int32
from funcy import print_durations
import open3d as o3d

"""
note: pcd format: float32
open3d format: float32[N,3] rgb (0,1)
"""


# @print_durations(unit='ms')
@vectorize([float32(float32),
            float32(int32)])
def intensity_to_color_pcd(intensity):
    if intensity < 30:
        green = int(intensity / 30 * 255)
        red = 0
        green = green & 255
        blue = 255
    elif intensity < 90:
        blue = int((90 - intensity) / (90 - 30) * 255) & 255
        red = 0
        green = 255
    elif intensity < 150:
        red = int((intensity - 90) / (150 - 90) * 255) & 255
        green = 255
        blue = 0
    else:
        green = int((255 - intensity) / (255 - 150) * 255) & 255
        red = 255
        blue = 0

    hex_color = np.uint32((red << 16) | (green << 8) | (blue << 0))

    # reinterpret_cast
    return hex_color.view(np.float32)


def color_pcd_to_color_o3d(pcd_color):
    pcd_color = pcd_color.view(np.uint32)
    red = (pcd_color >> 16) / 255.0
    green = ((pcd_color >> 8) & 255) / 255.0
    blue = (pcd_color & 255) / 255.0
    return np.hstack([red[:, None], green[:, None], blue[:, None]])


def intensity_to_color_o3d(intensity, is_normalized=False):
    if is_normalized:
        normalized_min = np.min(intensity)
        normalized_max = np.max(intensity)
    else:
        normalized_min = 0.0
        normalized_max = 255.0
    # [:,:3]: remove the alpha channel
    color_o3d = plt.get_cmap('jet')((intensity - normalized_min) / (normalized_max - normalized_min))[:, :3]
    return color_o3d


if __name__ == '__main__':
    def save_pcd(pointcloud, filename="sample.pcd"):
        with open(filename, 'w') as f:
            f.write("# .PCD v.7 - Point Cloud Data file format\n")
            f.write("VERSION .7\n")
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
            f.write("WIDTH {}\n".format(pointcloud.shape[0]))
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(pointcloud.shape[0]))
            f.write("DATA binary\n")
            pointcloud.tofile(f)


    pointcloud = np.load("data/stack_pointcloud.npy")[:, :4]
    pointcloud = np.array(pointcloud, dtype=np.float32)
    intensity = pointcloud[:, 3].copy()

    # achieve rviz/ros/pcd-format (float32) color
    color_pcd = intensity_to_color_pcd(pointcloud[:, 3])
    pointcloud[:, 3] = color_pcd
    save_pcd(pointcloud, filename="data/output.pcd")

    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])

    # color_o3d = color_pcd_to_color_o3d(color_pcd)
    color_o3d = intensity_to_color_o3d(intensity, is_normalized=True)
    pointcloud_o3d.colors = o3d.utility.Vector3dVector(color_o3d)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud_o3d)
    vis.get_render_option().point_size = 1
    vis.run()
    vis.destroy_window()

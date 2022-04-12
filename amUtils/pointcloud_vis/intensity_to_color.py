import numpy as np
from numba import vectorize, float32, int32
from funcy import print_durations
import open3d as o3d


# @print_durations(unit='ms')
@vectorize([float32(float32),
            float32(int32)])
def intensity2rgb_pcd(intensity):
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


def pcd_color_to_o3d_color(pcd_color):
    pcd_color = pcd_color.view(np.uint32)
    red = (pcd_color >> 16) / 255.0
    green = ((pcd_color >> 8) & 255) / 255.0
    blue = (pcd_color & 255) / 255.0
    return np.hstack([red[:, None], green[:, None], blue[:, None]])


def generate_pcd(pointcloud, filename="sample.pcd", format="binary"):
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
        if format == "ascii":
            f.write("DATA ascii\n")
            for i in range(pointcloud.shape[0]):
                f.write(
                    str(pointcloud[i][0]) + " " + str(pointcloud[i][1]) + " " + str(pointcloud[i][2]) + " " + str(
                        pointcloud[i][3]) + "\n")
        elif format == "binary":
            f.write("DATA binary\n")
            pointcloud.tofile(f)


if __name__ == '__main__':
    # 对点云的强度信息进行线性变换，得到颜色信息
    pointcloud = np.load("../data/livox_pointcloud.npy")
    # 得到rviz/ros/pcd的颜色信息(float32)
    pointcloud[:, 3] = intensity2rgb_pcd(pointcloud[:, 3])
    generate_pcd(pointcloud, filename="sample.pcd", format="binary")

    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
    # 得到open3d的颜色信息(float32(3))
    color_o3d = pcd_color_to_o3d_color(pointcloud[:, 3])
    pointcloud_o3d.colors = o3d.utility.Vector3dVector(color_o3d)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud_o3d)
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()

import cv2
import numpy as np
import open3d as o3d


class Calibration:

    def __init__(self):
        """
        TODO: remove hard coding
        the parameters abstract from calib_cam_to_cam.txt
        """
        self.fx = 9.597910e+02
        self.fy = 9.569251e+02
        self.cx = 6.960217e+02
        self.cy = 2.241806e+02
        T_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])
        T_03 = np.array([-4.731050e-01, 5.551470e-03, 5.250882e-03])
        self.baseline = np.linalg.norm(T_02 - T_03)

    def disparity2pointcloud(self, disparity_map, color_map):
        """
        Convert disparity map to point cloud.
        :param float[H,W] disparity_map:
        :param float fx: unit(pixel)
        :param float baseline: unit(m)
        """
        H = disparity_map.shape[0]
        W = disparity_map.shape[1]
        uy_img = (np.arange(H).reshape(-1, 1).repeat(W, axis=1))
        ux_img = (np.arange(W).reshape(1, -1).repeat(H, axis=0))

        # z = f * b / d = fx * b / d_img
        z_camera = self.fx * self.baseline / (disparity_map + 1e-6)
        x_camera = (ux_img - self.cx) / self.fx * z_camera
        y_camera = (uy_img - self.cy) / self.fy * z_camera

        # remove outlier
        mask = (z_camera > 0) & (z_camera < 100)

        blue = (color_map[..., 0][mask] / 255.0).reshape(-1, 1)
        green = (color_map[..., 1][mask] / 255.0).reshape(-1, 1)
        red = (color_map[..., 2][mask] / 255.0).reshape(-1, 1)
        color = np.hstack([red, green, blue])

        pointcloud = np.concatenate((x_camera[mask].reshape(-1, 1),
                                     y_camera[mask].reshape(-1, 1),
                                     z_camera[mask].reshape(-1, 1)), axis=-1)
        return pointcloud, color


if __name__ == '__main__':
    disparity_map = cv2.imread("data/disp_noc/000000_10.png", cv2.IMREAD_GRAYSCALE)
    color_map = cv2.imread("data/image_2/000000_10.png")

    cal = Calibration()
    pointcloud, color = cal.disparity2pointcloud(disparity_map, color_map)

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(color)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud_o3d)

    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()

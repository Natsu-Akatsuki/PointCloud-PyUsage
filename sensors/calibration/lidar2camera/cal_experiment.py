import cv2
import numpy as np
import open3d as o3d
from ampcl.calibration import Calibration
from matplotlib import pyplot as plt


def generate_project_img(colors, pts_img, undistor_img):
    colors[...] = colors[:, ::-1]  # RGB to BGR
    pc_mask = np.zeros_like(undistor_img)
    for (x, y), c in zip(pts_img, colors):
        cv2.circle(pc_mask, (x, y), 1, [c[0], c[1], c[2]], -1)
    intensity_img = cv2.addWeighted(undistor_img, 1, pc_mask, 0.5, 0)
    return intensity_img


class QualExperiment(Calibration):
    """
    perform qualitative experiment
    """

    def __init__(self):
        self.extri_matrix = None
        self.intri_matrix = None
        self.distor = None

    def paint_pointcloud(self, pointcloud, img, debug=False):

        # 对图像去畸变
        undistor_img = cv2.undistort(src=img, cameraMatrix=self.intri_matrix, distCoeffs=self.distor)
        pointcloud, pts_pixel = self.lidar_to_img_with_filter(pointcloud, undistor_img)

        # 索引颜色值
        color_np = undistor_img[np.int_(pts_pixel[:, 1]), np.int_(pts_pixel[:, 0])]

        if debug:
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
            color_np = color_np[:, ::-1] / 255
            point_cloud_o3d.colors = o3d.utility.Vector3dVector(color_np)
            o3d.visualization.draw_geometries([point_cloud_o3d], width=800, height=500)
        else:
            r = np.asarray(color_np[:, 2], dtype=np.uint32)
            g = np.asarray(color_np[:, 1], dtype=np.uint32)
            b = np.asarray(color_np[:, 0], dtype=np.uint32)
            rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
            rgb_arr.dtype = np.float32
            pointcloud[:, 3] = rgb_arr
            return pointcloud

    def project_pointcloud_to_img(self, pointcloud, img,
                                  fields=("intensity", "depth"),
                                  debug=False):
        """ 本部分也可以考虑畸变模型而采用OpenCV的projectPoints函数
        :param pointcloud:
        :param img:
        :param fields:
        :param debug:
        :return:

        """
        # 对图像去畸变
        undistor_img = cv2.undistort(src=img, cameraMatrix=self.intri_matrix, distCoeffs=self.distor)

        pointcloud, pts_img = self.lidar_to_img_with_filter(pointcloud, img)

        proj_img = [undistor_img]
        for field in fields:
            if field == "intensity":
                intensity = pointcloud[:, 3]
                normalized_min = 0.0
                normalized_max = 255.0

                # note: [:,:3]: remove the alpha channel
                colors = plt.get_cmap('jet')((intensity - normalized_min) /
                                             (normalized_max - normalized_min))[:, :3] * 255
                intensity_img = generate_project_img(colors, pts_img, undistor_img)
                proj_img.append(intensity_img)
            elif field == "depth":
                depth = np.linalg.norm(pointcloud[:, :3], axis=1)
                normalized_min = np.min(depth)
                normalized_max = np.max(depth)

                colors = plt.get_cmap('tab20')((depth - normalized_min) /
                                               (normalized_max - normalized_min))[:, :3] * 255
                depth_img = generate_project_img(colors, pts_img, undistor_img)
                proj_img.append(depth_img)
            else:
                raise ValueError("field {} is not supported".format(field))

        proj_img = np.hstack(proj_img)

        if debug:
            window_name = "undistorted"
            for field in fields:
                window_name = window_name + "-" + field
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 400)
            cv2.imshow(window_name, proj_img)
            cv2.waitKey(0)

        return proj_img

    def lidar_to_img_with_filter(self, pointcloud, img):
        # 将点云转换到像素平面上
        pts_img, z_camera = self.lidar_to_img(pointcloud[:, :3])
        # 只保留在相机前方的点
        mask = z_camera > 0
        pts_img = pts_img[mask]
        pointcloud = pointcloud[mask]
        # 只保留相机FOV内的点
        pts_img = np.floor(pts_img)
        mask = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img.shape[1]) & \
               (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img.shape[0])
        pts_img = pts_img[mask].astype(np.int32)
        pointcloud = pointcloud[mask]
        return pointcloud, pts_img

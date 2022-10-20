import numpy as np


class Calibration():
    def __init__(self):
        raise NotImplementedError("Please overwrite the init function")

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_camera_points(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_camera: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_camera = np.dot(pts_lidar_hom, self.extri_matrix.T)[:, 0:3]
        return pts_camera

    def camera_to_img_points(self, pts_camera, image_shape=None):
        """
        :param pts_camera: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_camera = np.dot(pts_camera, self.intri_matrix.T)
        pts_camera_depth = pts_camera[:, 2]
        pts_img = (pts_camera[:, 0:2].T / pts_camera[:, 2]).T  # (N, 2)

        if image_shape is not None:
            pts_img[:, 0] = np.clip(pts_img[:, 0], a_min=0, a_max=image_shape[1] - 1)
            pts_img[:, 1] = np.clip(pts_img[:, 1], a_min=0, a_max=image_shape[0] - 1)

        return pts_img, pts_camera_depth

    def lidar_to_img(self, pts_lidar, image_shape=None):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_camera = self.lidar_to_camera_points(pts_lidar)
        pts_img, pts_camera_depth = self.camera_to_img_points(pts_camera, image_shape=image_shape)
        return pts_img, pts_camera_depth

    def corners3d_lidar_to_boxes2d_image(self, corners3d_lidar, image_shape=None):
        """
        :param corners3d: (8, 3) corners in lidar coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8, 2) [xi, yi] in rgb coordinate
        """

        img_pts, _ = self.lidar_to_img(corners3d_lidar, image_shape)

        x, y = img_pts[:, 0], img_pts[:, 1]
        x1, y1 = np.min(x), np.min(y)
        x2, y2 = np.max(x), np.max(y)

        boxes2d_image = np.stack((x1, y1, x2, y2))
        boxes_corner = np.stack((x, y), axis=1)

        if image_shape is not None:
            boxes2d_image[0] = np.clip(boxes2d_image[0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[1] = np.clip(boxes2d_image[1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[2] = np.clip(boxes2d_image[2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[3] = np.clip(boxes2d_image[3], a_min=0, a_max=image_shape[0] - 1)

        return boxes2d_image, boxes_corner

    # lidar -> camera
    def lidar_to_camera_lwh3d(self, boxes3d_lidar):
        """
        Args:
            boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            calib: {obj}
        Returns:
            boxes3d_camera: (N, 7) [x, y, z, l, w, h, r] in camera coord
        """
        xyz_lidar = boxes3d_lidar[:, 0:3]
        l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = self.lidar_to_camera_points(xyz_lidar)
        r = -r - np.pi / 2
        return np.concatenate([xyz_cam, l, w, h, r], axis=-1)

    # camera -> lidar
    def camera_to_lidar_points(self, pts_camera):
        """
        Args:pts_camera
            pts_camera: (N, 3)
        Returns:
            pts_lidar: (N, 3)
        """
        pts_camera_hom = self.cart_to_hom(pts_camera)
        pts_lidar = (pts_camera_hom @ np.linalg.inv(self.extri_matrix.T))[:, 0:3]
        return pts_lidar


class LivoxCalibration(Calibration):
    def __init__(self, calib_file):

        if not isinstance(calib_file, dict):
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.intri_matrix = calib['intri_matrix']
        self.distor = calib['distor']
        self.extri_matrix = calib['extri_matrix']

    @staticmethod
    def get_calib_from_file(calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        intri_matrix = np.vstack((np.asarray(lines[1].strip().split(), dtype=np.float32),
                                  np.asarray(lines[2].strip().split(), dtype=np.float32),
                                  np.asarray(lines[3].strip().split(), dtype=np.float32),
                                  ))
        distor = np.asarray(lines[6].strip().split(), dtype=np.float32)

        extri_matrix = np.vstack((np.asarray(lines[9].strip().split(), dtype=np.float32),
                                  np.asarray(lines[10].strip().split(), dtype=np.float32),
                                  np.asarray(lines[11].strip().split(), dtype=np.float32),
                                  np.asarray(lines[12].strip().split(), dtype=np.float32),
                                  ))

        return {'intri_matrix': intri_matrix,  # (3,3)
                'distor': distor,  # 4
                'extri_matrix': extri_matrix  # (4,4)
                }

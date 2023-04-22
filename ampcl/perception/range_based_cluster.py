import cv2
import numpy as np
from matplotlib import pyplot as plt


def xyz_to_yaw_pitch_range(pc_np):
    assert len(pc_np.shape) == 2, "The pointcloud should be [N, ...] e.g. [N, 3]"
    x = pc_np[:, 0]
    y = pc_np[:, 1]
    z = pc_np[:, 2]

    # xy_distance = np.linalg.norm(pointcloud[:, :2], axis=1, ord=2)
    range = np.linalg.norm(pc_np[:, :3], axis=1, ord=2)

    # 调整起点位置
    #         ^ 0                       ^ 0                          ^ 0
    #         |                         |                            |
    #         |            取反          |            +180°           |
    #   <—— —— —— ——       -->    <—— —— —— ——        -->      <—— —— —— ——
    #         |                         |                            |
    #         |                         |                            |
    #  （-180）| （180）          （-180）| （180）                （0） | （360）
    yaw = np.rad2deg(-np.arctan2(y, x)) + 180.0
    # note: pitch ≠ inclination angle
    pitch = np.rad2deg(np.arcsin(z / range))

    print(f"The range of azimuth: ({np.min(yaw)}, {np.max(yaw)})")
    print(f"The range of pitch: ({np.min(pitch)}, {np.max(pitch)})")

    return yaw, pitch, range


def yaw_pitch_to_img_xy(yaw, pitch,
                        img_width=1024, img_height=64,
                        horizon_offset=0.0, vertical_offset=-24.9):
    img_x_resolution = (img_width / 360.0)
    img_y_resolution = img_height / (24.9 + 2.0)

    img_x = (yaw - horizon_offset) * img_x_resolution
    img_y = img_height - (pitch - vertical_offset) * img_y_resolution

    # 四舍五入+截断
    img_x = np.clip(np.floor(img_x), 0, img_width - 1).astype(np.int32)  # in [0, W-1]
    img_y = np.clip(np.floor(img_y), 0, img_height - 1).astype(np.int32)  # in [0, H-1]

    return img_x, img_y


def create_range_img(pc_np,
                     img_width=1800, img_height=64,
                     debug=False
                     ):
    yaw, pitch, range = xyz_to_yaw_pitch_range(pc_np)
    img_x, img_y = yaw_pitch_to_img_xy(yaw, pitch, img_width=img_width, img_height=img_height)

    pixel = range
    range_img = np.full((img_height, img_width), 0, dtype=np.float32)
    pc_index_img = np.full((img_height, img_width), -1, dtype=np.int32)
    range_img[img_y, img_x] = pixel
    pc_index_img[img_y, img_x] = np.arange(len(pc_np))

    if debug:
        pixel = pc_np[:, 3] * 255  # *255 for KITTI
        img1 = np.full((img_height, img_width), 0, dtype=np.uint8)
        img1[img_y, img_x] = pixel

        pixel = range / 100 * 255
        img2 = np.full((img_height, img_width), 0, dtype=np.uint8)
        img2[img_y, img_x] = pixel

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 2), dpi=400)
        ax1.imshow(img1)
        ax1.set_title("Intensity image")
        ax2.imshow(img2)
        ax2.set_title("Range image")
        plt.show()

        img1 = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
        cv2.namedWindow(f"Intensity image", cv2.WINDOW_FREERATIO)
        cv2.imshow(f"Intensity image", img1)
        cv2.waitKey(0)

    return range_img, pc_index_img


class RangeImgCluster:
    """
    velodyne-16: horizon_res=0.2, vertical_res=2, threshold_h=60, threshold_v=60, valid_point_num=30
    velodyne-64: horizon_res=0.15, vertical_res=0.4, threshold_h=60, threshold_v=60, valid_point_num=30
    """

    def __init__(self, horizon_res=0.2, vertical_res=2,
                 threshold_h=60, threshold_v=60, valid_point_num=30,
                 img_width=1800, img_height=64,
                 ):

        self.neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.horizon_res = np.deg2rad(horizon_res)
        self.vertical_res = np.deg2rad(vertical_res)
        self.threshold_h = np.deg2rad(threshold_h)
        self.threshold_v = np.deg2rad(threshold_v)
        self.valid_point_num = valid_point_num
        self.img_width = img_width
        self.img_height = img_height

    def cluster(self, pc_np, debug=False):
        range_img, index_img = create_range_img(pc_np, img_width=self.img_width,
                                                img_height=self.img_height, debug=debug)

        self.nrow = range_img.shape[0]
        self.ncol = range_img.shape[1]
        label_img = np.zeros_like(range_img, dtype=np.int32)

        label = 0
        cluster_idx_list = []
        for i in range(0, self.nrow):
            for j in range(self.ncol):
                if label_img[i, j] == 0:
                    label += 1
                    cluster_idx = self.label_component_bfs(i, j, label, label_img, range_img, index_img)
                    if len(cluster_idx) > self.valid_point_num:
                        cluster_idx_list.append(cluster_idx)

        return cluster_idx_list

    def label_component_bfs(self, i, j, label, label_img, range_img, index_img):

        cluster_idx = [index_img[i, j]]
        queue = [[i, j]]
        while len(queue) != 0:
            target = queue.pop(0)
            target_i = target[0]
            target_j = target[1]

            if label_img[target_i, target_j] == 0:
                label_img[target_i, target_j] = label

            for r, c in self.neighbors:
                neighbor_i = target_i + r
                neighbor_j = target_j + c

                # margin case
                if neighbor_i < 0 or neighbor_i >= self.nrow:
                    continue
                if neighbor_j < 0:
                    neighbor_j = self.ncol - 1
                if neighbor_j >= self.ncol:
                    neighbor_j = 0
                if label_img[neighbor_i, neighbor_j] != 0:
                    continue

                alpha = self.horizon_res
                threshold_ = self.threshold_h
                if c == 0:
                    alpha = self.vertical_res
                    threshold_ = self.threshold_v

                range_neighbor = range_img[neighbor_i, neighbor_j]
                range_target = range_img[target_i, target_j]

                d1 = max(range_neighbor, range_target)
                d2 = min(range_neighbor, range_target)

                threshold = np.arctan2(
                    (d2 * np.sin(alpha)),
                    (d1 - d2 * np.cos(alpha)))

                if threshold > threshold_:
                    label_img[neighbor_i, neighbor_j] = label
                    if [neighbor_i, neighbor_j] in queue:
                        pass
                    else:
                        queue.append([neighbor_i, neighbor_j])
                        cluster_idx.append(index_img[neighbor_i, neighbor_j])

        return cluster_idx

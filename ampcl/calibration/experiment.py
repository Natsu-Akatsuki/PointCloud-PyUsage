import cv2
import numpy as np
from matplotlib import pyplot as plt

from .calibration import lidar_to_pixel
from .. import visualization
from ..ros import np_to_pointcloud2
from ..visualization import o3d_viewer_from_pointcloud


def generate_project_img(value, pixel_coord, undistor_img,
                         normalized_max=255, normalized_min=0,
                         color_mode="jet"):
    colors = plt.get_cmap(color_mode)((value - normalized_min) /
                                      (normalized_max - normalized_min))[:, :3] * 255  # remove alpha channel

    colors[...] = colors[:, ::-1]  # RGB to BGR
    pc_mask = np.zeros_like(undistor_img)
    for (x, y), c in zip(pixel_coord, colors):
        x, y = int(x), int(y)
        cv2.circle(pc_mask, (x, y), 2, c, -1)
    mask_img = cv2.addWeighted(undistor_img, 1, pc_mask, 0.8, 0)
    return mask_img


def paint_pointcloud(pc_np, img, cal_info, debug=False, publisher=None, header=None):
    distor = cal_info["distor"]
    intri_mat = cal_info["intri_mat"]
    extri_mat = cal_info["extri_mat"]
    undistor_img = cv2.undistort(src=img, cameraMatrix=intri_mat, distCoeffs=distor)
    img_shape = undistor_img.shape
    pixel_coord, _, mask = lidar_to_pixel(pc_np, cal_info, img_shape, use_mask=True)

    pc_filtered = pc_np[mask]

    # 索引颜色值
    color3u8 = undistor_img[np.int_(pixel_coord[:, 1]), np.int_(pixel_coord[:, 0])]

    if debug:
        color3f = color3u8[:, ::-1] / 255
        o3d_viewer_from_pointcloud(pc_filtered, colors=color3f, width=800, height=500)

    if publisher is not None:
        color1f = visualization.color3u8_to_color1f(color3u8)
        pc_np_with_color = np.hstack([pc_filtered, color1f])
        pc2_msg = np_to_pointcloud2(pc_np_with_color, header, field="xyzirgb")
        publisher.publish(pc2_msg)


def project_pc_to_img(pc_np, img,
                      cal_info,
                      fields=("intensity", "range"),
                      debug=False):
    distor = cal_info["distor"]
    intri_mat = cal_info["intri_mat"]
    extri_mat = cal_info["extri_mat"]

    undistor_img = cv2.undistort(src=img, cameraMatrix=intri_mat, distCoeffs=distor)
    img_shape = undistor_img.shape
    pixel_coord, _, mask = lidar_to_pixel(pc_np, cal_info, img_shape, use_mask=True)

    proj_img = [undistor_img]
    for field in fields:
        if field == "intensity":
            intensity = pc_np[:, 3]
            intensity_img = generate_project_img(intensity, pixel_coord, undistor_img, normalized_max=255,
                                                 normalized_min=0, color_mode="jet")
            proj_img.append(intensity_img)
        elif field == "range":
            range = np.linalg.norm(pc_np[:, :3], axis=1)
            normalized_min = np.min(range)
            normalized_max = np.max(range)
            range_img = generate_project_img(range, pixel_coord, undistor_img, normalized_max=normalized_max,
                                             normalized_min=normalized_min, color_mode="tab20")
            proj_img.append(range_img)
        elif field == "black":
            color3u8 = undistor_img[np.int_(pixel_coord[:, 1]), np.int_(pixel_coord[:, 0])]
            pc_mask = np.zeros_like(undistor_img)
            for (x, y), c in zip(pixel_coord, color3u8):
                x, y = int(x), int(y)
                c = c.tolist()
                cv2.circle(pc_mask, (x, y), 2, c, -1)
            proj_img.append(pc_mask)
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

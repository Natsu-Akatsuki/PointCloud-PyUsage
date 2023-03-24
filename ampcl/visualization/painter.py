import numpy as np
import cv2


def id_to_color(cls_id):
    color_maps = {-1: [0.5, 0.5, 0.5],
                  0: [1.0, 0.0, 0.0],
                  1: [0.0, 1.0, 0.0],  # Vehicle
                  2: [0.0, 0.0, 1.0],  # Pedestrian
                  3: [1.0, 0.8, 0.0],  # Cyclist
                  4: [1.0, 0.0, 1.0]
                  }
    if cls_id not in color_maps.keys():
        return [0.5, 1.0, 0.5]

    return color_maps[cls_id]


def paint_box2d_on_img(img, box2d, cls_idx=None):
    img = img.copy()
    for i in range(box2d.shape[0]):
        box = box2d[i]
        if cls_idx is None:
            box_color = (0, 0, 255)
        else:
            box_color = (np.asarray(id_to_color(cls_idx[i])) * 255)[::-1]
            box_color = tuple([int(x) for x in box_color])
        box = tuple([int(x) for x in box[:4]])
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), box_color, 2)

    return img


def paint_box3d_on_img(img, corner_cam, use_8_corners=True, cls_idx=None):
    """
        4-------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2

    :param img:
    :param corner_cam: (N, 8, 3) 相机系的8个角点
    :param use_8_corners: 使用8角点还是4角点
    :param cls_idx:
    :return:
    """
    img = img.copy()
    for i in range(corner_cam.shape[0]):
        if cls_idx is None:
            box_color = (0, 0, 255)
        else:
            box_color = (np.asarray(id_to_color(cls_idx[i])) * 255)[::-1]
            box_color = tuple([int(x) for x in box_color])

        bbx_3d = corner_cam[i]
        if use_8_corners:
            # draw the 4 vertical lines
            cv2.line(img, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[2].astype(int)), tuple(bbx_3d[6].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[3].astype(int)), tuple(bbx_3d[7].astype(int)), box_color, 2)

            # draw the 4 horizontal lines
            cv2.line(img, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[1].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[2].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[2].astype(int)), tuple(bbx_3d[3].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[3].astype(int)), tuple(bbx_3d[0].astype(int)), box_color, 2)

            # draw the 4 horizontal lines
            cv2.line(img, tuple(bbx_3d[4].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[5].astype(int)), tuple(bbx_3d[6].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[6].astype(int)), tuple(bbx_3d[7].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[7].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)

            cv2.line(img, tuple(bbx_3d[0].astype(int)), tuple(bbx_3d[5].astype(int)), box_color, 2)
            cv2.line(img, tuple(bbx_3d[1].astype(int)), tuple(bbx_3d[4].astype(int)), box_color, 2)
        else:
            x1, y1 = int(np.min(bbx_3d, axis=0)), int(np.min(bbx_3d, axis=1))
            x2, y2 = int(np.max(bbx_3d, axis=0)), int(np.max(bbx_3d, axis=1))
            image_shape = img.shape
            x1 = np.clip(x1, a_min=0, a_max=image_shape[1] - 1)
            y1 = np.clip(y1, a_min=0, a_max=image_shape[0] - 1)
            x2 = np.clip(x2, a_min=0, a_max=image_shape[1] - 1)
            y2 = np.clip(y2, a_min=0, a_max=image_shape[0] - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    return img

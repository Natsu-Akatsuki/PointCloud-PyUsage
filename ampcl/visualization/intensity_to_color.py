import numpy as np
from matplotlib import pyplot as plt
from numba import float32, int32, vectorize

"""
note:
rgb pcd / rviz format: float32[N] float_rgb [_, r, g, b] (0-255), includes 3 single-byte values with range from 0 to 255
open3d format: float32[N,3] rgb (0,1)
"""


@vectorize([float32(float32),
            float32(int32)])
def intensity_to_color(intensity):
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
    # hex_color.dtype = np.float32
    return hex_color.view(np.float32)


def intensity_to_color_o3d(intensity, is_normalized=True):
    if is_normalized:
        intensity = np.int32(intensity * 255)

    color_channel = intensity_to_color(intensity).view(np.uint32)

    # multi-stage pseudo-colorize
    red = (color_channel >> 16) / 255.0
    green = ((color_channel >> 8) & 255) / 255.0
    blue = (color_channel & 255) / 255.0
    color_o3d = np.stack([red, green, blue], axis=1)

    # single-stage pseudo-colorize
    # if is_normalized:
    #     normalized_min = np.min(intensity)
    #     normalized_max = np.max(intensity)
    # else:
    #     normalized_min = 0.0
    #     normalized_max = 255.0
    # [:,:3]: remove the alpha channel
    # color_o3d = plt.get_cmap('jet')((intensity - normalized_min) / (normalized_max - normalized_min))[:, :3]

    return color_o3d


def intensity_to_color_pcd(intensity):
    color_channel = intensity_to_color(intensity).view(np.uint32)
    return color_channel

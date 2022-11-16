import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def passthrough_filter(pointcloud: object, limit_range: object = list):
    """
    :param pointcloud:
    :param limit_range: [x_min, x_max, y_min, y_max, z_min, z_max]
    :return:
    """
    mask = np.full(pointcloud.shape[0], False)
    if limit_range is None:
        return mask
    for i in range(pointcloud.shape[0]):
        if (pointcloud[i, 0] >= limit_range[0]) & (pointcloud[i, 0] <= limit_range[1]) \
                & (pointcloud[i, 1] >= limit_range[2]) & (pointcloud[i, 1] <= limit_range[3]) \
                & (pointcloud[i, 2] >= limit_range[4]) & (pointcloud[i, 2] <= limit_range[5]):
            mask[i] = True
    return mask

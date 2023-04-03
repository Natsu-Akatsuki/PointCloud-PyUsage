import numpy as np


def filter_livox_noise_point(pc_np, remove_nan=True):
    """移除无效的激光点，Livox的激光点一般设为(0, 0, 0)
    """
    if remove_nan:
        nan_indices = np.isnan(pc_np).any(axis=1)
        pc_np = pc_np[~nan_indices]

    # 移除（0，0，0）点
    pc_np = pc_np[~(pc_np[:, 0] < 0.01)]
    return pc_np

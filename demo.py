from funcy import print_durations

from ampcl.io import load_pointcloud, save_pointcloud


def io_demo():
    # >>> import usage >>>
    pointcloud = load_pointcloud("ampcl/data/pointcloud.pcd")

    # >>> export usage >>>
    save_pointcloud(pointcloud, "pointcloud.pcd")


def visualization_demo():
    # >>> visualize >>>
    from ampcl.visualization import o3d_viewer_from_pointcloud
    pointcloud = load_pointcloud("ampcl/data/pointcloud.pcd")
    o3d_viewer_from_pointcloud(pointcloud)


def downsampe_demo():
    from ampcl.io import load_pointcloud
    from ampcl.filter import passthrough_filter, c_voxel_filter

    pointcloud = load_pointcloud("ampcl/data/pointcloud.pcd")
    pointcloud = pointcloud[passthrough_filter(pointcloud, limit_range=(0, 40, -40, 40, -2, 2))]

    with print_durations("ms"):
        c_voxel_filter(pointcloud, voxel_size=[0.01, 0.01, 0.01])


if __name__ == "__main__":
    downsampe_demo()

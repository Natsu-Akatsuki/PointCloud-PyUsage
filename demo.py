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
    pointcloud = load_pointcloud("ampcl/data/pointcloud_ascii.pcd")
    o3d_viewer_from_pointcloud(pointcloud)


if __name__ == "__main__":
    visualization_demo()

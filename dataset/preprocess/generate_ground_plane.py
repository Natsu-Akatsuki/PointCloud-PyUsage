from pathlib import Path
from ampcl.perception.ground_segmentation import ground_segmentation_ransac
from ampcl.filter import c_ransac_plane_fitting
from ampcl.filter import passthrough_filter
from ampcl.io import load_pointcloud
from ampcl.visualization import o3d_viewer_from_pointcloud
from ampcl.calibration import calibration_livox
from tqdm import tqdm
import numpy as np
import open3d as o3d

plane_dir = Path("planes")
plane_dir.mkdir(exist_ok=True)

horizon_dir = Path.home() / Path("mnt/dataset/livox_dataset/training/horizon")
pointcloud_paths = sorted(list(horizon_dir.iterdir()))


def plane_model_transformation(plane_model, transformation):
    """ 将A系下的平面模型转换到B系下
    :param plane_model: (4,)
    :param transformation: (4, 4) 将A系的点转换B系的变换矩阵
    :return:
    """
    plane_model = plane_model.copy()
    plane_model = np.linalg.inv(transformation).T @ plane_model
    return plane_model


cal = calibration_livox.LivoxCalibration(Path.home() / Path("mnt/dataset/livox_dataset/training/calib/config.txt"))

for i, pointcloud_path in enumerate(tqdm(pointcloud_paths)):
    index_stem = Path(pointcloud_path).stem
    pointcloud_path = str(Path(horizon_dir / Path(pointcloud_path)))
    pointcloud_np = load_pointcloud(pointcloud_path)
    limit_range = (0, 50, -10, 10, -2.5, -1.0)
    plane_model, _ = ground_segmentation_ransac(pointcloud_np, limit_range, distance_threshold=0.5, debug=False)

    plane_model = plane_model_transformation(plane_model, cal.extri_matrix)

    if 0:
        pointcloud_np = cal.lidar_to_camera_points(pointcloud_np[:, :3])
        dis = np.fabs(pointcloud_np[:, :3] @ plane_model[:3] + plane_model[3])
        indices = np.arange(pointcloud_np.shape[0])
        ground_idx = indices[dis < 0.5]
        non_ground_idx = indices[dis >= 0.5]
        ground_point_o3d = o3d_viewer_from_pointcloud(pointcloud_np[ground_idx], is_show=False)
        ground_point_o3d.paint_uniform_color([1, 0, 0])
        non_ground_point_o3d = o3d_viewer_from_pointcloud(pointcloud_np[non_ground_idx], is_show=False)
        non_ground_point_o3d.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([ground_point_o3d, non_ground_point_o3d])

    content = "# Plane\n"
    content += "Width 4\n"
    content += "Height 1\n"
    content += f"{plane_model[0]} {plane_model[1]} {plane_model[2]} {plane_model[3]}"

    with open(f"{plane_dir}/{index_stem}.txt", "w") as file:
        file.write(content)

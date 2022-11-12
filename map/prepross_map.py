import open3d as o3d
from ampcl.io import load_pointcloud

if __name__ == '__main__':
    file_name = "/home/helios/mnt/Dataset/SurfMap.pcd"
    print(f"[IO] load file {file_name}")
    pc_np = load_pointcloud(file_name)
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pc_np[:, 0:3])

    downsampled_pc_o3d = pointcloud_o3d.voxel_down_sample(voxel_size=0.2)
    print(f"[downsample] point nums: {len(downsampled_pc_o3d.points)} -> {len(pointcloud_o3d.points)}")

    filtered_pc_o3d, inliers = downsampled_pc_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[filter] point nums: {len(downsampled_pc_o3d.points)} -> {len(filtered_pc_o3d.points)}")

    o3d.visualization.draw_geometries([filtered_pc_o3d])

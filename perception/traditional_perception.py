import argparse
import os

import open3d as o3d
from ampcl.io import load_pointcloud
from ampcl.perception import cEuclideanCluster, ground_segmentation_gpf
from ampcl.ros.marker import instance_id_to_color
from funcy import print_durations

from shape_estimation import convex_hull


@print_durations()
def ground_segmentation(pc_np, o3d_objs, debug=False):
    # 算法一：GPF
    ground_mask = ground_segmentation_gpf(pc_np, debug=False)

    # 算法二：RANSAC
    # limit_range = (-50, 50, -50, 50, -2.5, 0)
    # plane_model, ground_mask = ground_segmentation_ransac(pc_np, limit_range,
    #                                                 distance_threshold=0.3,
    #                                                 max_iterations=10,
    #                                                 debug=False)

    if debug:
        ground_pc_o3d = o3d.geometry.PointCloud()
        ground_pc_o3d.points = o3d.utility.Vector3dVector(pc_np[ground_mask][:, :3])
        ground_pc_o3d.paint_uniform_color([0, 0, 1])
        o3d_objs.append(ground_pc_o3d)

    non_ground_pc = pc_np[~ground_mask]
    return non_ground_pc


@print_durations()
def shape_estimation(clusters, debug=False, o3d_objs=None):
    if o3d_objs is None:
        o3d_objs = list()

    convex_model = convex_hull.ShapeModel()

    cluster_shapes = []
    for i, cluster in enumerate(clusters):
        cluster_shape = convex_model.estimate(cluster[:, :3])
        cluster_shapes.append(cluster_shape)

        if debug:
            o3d_objs.append(convex_model.generate_polygon_o3d())
            o3d_objs.append(convex_model.generate_centroid_o3d())

    return cluster_shapes


@print_durations()
def cluster_segmentation(non_ground_pc, o3d_objs, debug=False):
    cluster_idx_list = cEuclideanCluster(non_ground_pc, tolerance=0.5, min_size=20, max_size=30000)
    cluster_list = []
    for i, cluster_idx in enumerate(cluster_idx_list):
        cluster = non_ground_pc[cluster_idx]
        cluster_list.append(cluster)
        if debug:
            cluster_o3d = o3d.geometry.PointCloud()
            cluster_o3d.points = o3d.utility.Vector3dVector(cluster[:, :3])
            cluster_o3d.paint_uniform_color(instance_id_to_color(i))
            o3d_objs.append(cluster_o3d)
    return cluster_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pc_dir', type=str, default='/home/helios/mnt/dataset/Kitti/object/training/velodyne/',
                        help='directory of point cloud files')
    args = parser.parse_args()

    pc_file_list = sorted(os.listdir(args.pc_dir))
    for idx, pc_file in enumerate(pc_file_list):
        print(f"pc_file: {pc_file}")
        pc_file = os.path.join(args.pc_dir, pc_file)
        pc_np = load_pointcloud(pc_file)

        o3d_objs = []
        # 步骤一：地面分割
        non_ground_pc = ground_segmentation(pc_np, o3d_objs, debug=True)

        # 步骤二：聚类分割
        cluster_list = cluster_segmentation(non_ground_pc, o3d_objs, debug=True)

        # 步骤三：形状估测
        shape_estimation(cluster_list, debug=True, o3d_objs=o3d_objs)

        # 步骤四：可视化
        o3d.visualization.draw_geometries(o3d_objs,
                                          width=900, height=600,
                                          zoom=0.14,
                                          front=[-0.72, -0.18, 0.66],
                                          lookat=[7.12, 4.78, -0.95],
                                          up=[0.67, 0.05, 0.75])

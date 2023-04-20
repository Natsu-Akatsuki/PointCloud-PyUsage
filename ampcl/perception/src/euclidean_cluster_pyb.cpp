
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

#include <Eigen/SVD>

namespace py = pybind11;
typedef pcl::PointXYZ PointT;
const int FIELD_NUM = 3;


std::vector<std::vector<int>> cEuclideanCluster(const py::array_t<float> &input,
                                                float tolerance,
                                                int min_size, int max_size) {
  auto input_ref = input.unchecked<2>();
  pcl::PointCloud<PointT>::Ptr
    pc_pcl_ptr(new pcl::PointCloud<PointT>(input_ref.shape(0), 1));
  for (int i = 0; i < input_ref.shape(0); ++i) {
    pc_pcl_ptr->points[i].x = input_ref(i, 0);
    pc_pcl_ptr->points[i].y = input_ref(i, 1);
    pc_pcl_ptr->points[i].z = input_ref(i, 2);
  }


  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<PointT>);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(tolerance);
  ec.setMinClusterSize(min_size);
  ec.setMaxClusterSize(max_size);
  ec.setSearchMethod(tree);

  ec.setInputCloud(pc_pcl_ptr);
  ec.extract(cluster_indices);

  std::vector<std::vector<int>> cluster_idx_list(cluster_indices.size());

  for (int j = 0; j < cluster_indices.size(); ++j) {
    for (const auto &idx: cluster_indices[j].indices) {
      cluster_idx_list[j].push_back(idx);
    }
  }

  return cluster_idx_list;
}


PYBIND11_MODULE(euclidean_cluster_pyb, m) {
  m.doc() = "euclidean cluster module for pointcloud";
  m.def("cEuclideanCluster", &cEuclideanCluster,
        "filter pointcloud",
        "input"_a,
        "tolerance"_a = 0.5,
        "min_size"_a = 20,
        "max_size"_a = 25000);
}
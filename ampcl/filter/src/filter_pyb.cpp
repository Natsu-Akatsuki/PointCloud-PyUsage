#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;
typedef pcl::PointXYZI PointT;
const int FIELD_NUM = 4;

py::array_t<float> cVoxelFilter(const py::array_t<float> &input,
                                const std::vector<float> &voxel_size,
                                const std::string &mode = "mean") {

  auto input_ref = input.unchecked<2>();
  pcl::PointCloud<PointT>::Ptr pointcloud(new pcl::PointCloud<PointT>(input_ref.shape(0), 1));
  for (int i = 0; i < input_ref.shape(0); ++i) {
    pointcloud->points[i].x = input_ref(i, 0);
    pointcloud->points[i].y = input_ref(i, 1);
    pointcloud->points[i].z = input_ref(i, 2);
    pointcloud->points[i].intensity = input_ref(i, 3);
  }

  if (mode == "mean") {
    pcl::VoxelGrid<PointT> filter;
    filter.setInputCloud(pointcloud);
    filter.setLeafSize(voxel_size[0], voxel_size[1], voxel_size[2]);
    filter.filter(*pointcloud);
  } else if (mode == "uniform") {
    pcl::UniformSampling<PointT> filter;
    filter.setInputCloud(pointcloud);
    filter.setRadiusSearch(voxel_size[0]);
    filter.filter(*pointcloud);
  }

  unsigned int pc_size = pointcloud->width * pointcloud->height;
  py::array_t<float, py::array::c_style> pointcloud_output({(const int) pc_size, FIELD_NUM});

  auto pointcloud_proxy = pointcloud_output.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < pointcloud_proxy.shape(0); i++) {
    pointcloud_proxy(i, 0) = pointcloud->points[i].x;
    pointcloud_proxy(i, 1) = pointcloud->points[i].y;
    pointcloud_proxy(i, 2) = pointcloud->points[i].z;
    pointcloud_proxy(i, 3) = pointcloud->points[i].intensity;
  }

  return pointcloud_output;
}

PYBIND11_MODULE(filter_pyb, m) {
  m.doc() = "an filter module for pointcloud";
  m.def("cVoxelFilter", &cVoxelFilter, "filter pointcloud", "input"_a, "voxel_size"_a, "mode"_a = "mean");
};

#include <iostream>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace py::literals;

// 有关读取的点云类型
typedef pcl::PointXYZI PointType;
#define FIELD_NUM 4

int save_pcd_file(std::string &file_name, const py::array_t<float> &input) {

  std::cout << "saving file: " << file_name << std::endl;
  auto input_ref = input.unchecked<2>();
  pcl::PointCloud<PointType>::Ptr pointcloud(
      new pcl::PointCloud<PointType>(input_ref.shape(0), 1));
  for (int i = 0; i < input_ref.shape(0); ++i) {
    pointcloud->points[i].x = input_ref(i, 0);
    pointcloud->points[i].y = input_ref(i, 1);
    pointcloud->points[i].z = input_ref(i, 2);
    pointcloud->points[i].intensity = input_ref(i, 3);
  }
  if (pcl::io::savePCDFile<PointType>(file_name, *pointcloud, true) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return (-1);
  }
  return 0;
}

PYBIND11_MODULE(save_pcd_file_pcl, m) {
  m.doc() = "save pcd pointcloud";
  m.def("save_pcd_file", &save_pcd_file, "save pcd file", "file_name"_a,
        "input"_a);
};
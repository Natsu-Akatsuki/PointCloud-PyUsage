#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

// 有关读取的点云类型
typedef pcl::PointXYZI PointType;
#define FIELD_NUM 4

py::array_t<float> cLoadPCD(std::string &file_name, bool remove_nan) {
  pcl::PointCloud<PointType>::Ptr pointcloud(new pcl::PointCloud<PointType>);
  std::cout << "[cIO] Loading file: " << file_name << std::endl;
  if (pcl::io::loadPCDFile<PointType>(file_name, *pointcloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file\n");
  }
  if (remove_nan) {
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pointcloud, *pointcloud, indices);
  }

  unsigned int pc_size = pointcloud->width * pointcloud->height;
  py::array_t<float, py::array::c_style> pointcloud_output({(const int) pc_size, FIELD_NUM});

  // note: 用这种方法可以使用()的赋值方法，模板实参2表示维度
  auto pointcloud_proxy = pointcloud_output.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < pointcloud_proxy.shape(0); i++) {
    pointcloud_proxy(i, 0) = pointcloud->points[i].x;
    pointcloud_proxy(i, 1) = pointcloud->points[i].y;
    pointcloud_proxy(i, 2) = pointcloud->points[i].z;
    pointcloud_proxy(i, 3) = pointcloud->points[i].intensity;
  }

  return pointcloud_output;
}

int savePCD(std::string &file_name, const py::array_t<float> &input) {

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
    PCL_ERROR("Couldn't save file\n");
    return (-1);
  }
  return 0;
}

PYBIND11_MODULE(io_pyb, m) {
  m.doc() = "an IO module for PCD file";
  m.def("cLoadPCD", &cLoadPCD, "load pcd file", "file_name"_a, "remove_nan"_a = true);
  m.def("csavePCD", &savePCD, "save pcd file", "file_name"_a, "input"_a);
};

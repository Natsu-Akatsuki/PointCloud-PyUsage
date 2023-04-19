#include <pcl/point_types.h>
#include <pcl/common/pca.h>

#include <iostream>

#include <pcl/segmentation/sac_segmentation.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace py::literals;

#include <Eigen/SVD>

namespace py = pybind11;
typedef pcl::PointXYZ PointT;
const int FIELD_NUM = 3;

class GPF {
public:
    explicit GPF(float sensor_height = 1.73, float inlier_threshold = 0.3, int iter_num = 3,
                 int num_lpr = 250, float seed_height_offset = 1.2)
      : sensor_height_(sensor_height), inlier_threshold_(inlier_threshold), iter_num_(iter_num),
        num_lpr_(num_lpr), seed_height_offset_(seed_height_offset) {}

    py::array_t<bool> apply(const py::array_t<float> &input) {

      auto input_ref = input.unchecked<2>();
      pcl::PointCloud<PointT>::Ptr pointcloud(new pcl::PointCloud<PointT>(input_ref.shape(0), 1));
      for (auto i = 0; i < input_ref.shape(0); ++i) {
        pointcloud->points[i].x = input_ref(i, 0);
        pointcloud->points[i].y = input_ref(i, 1);
        pointcloud->points[i].z = input_ref(i, 2);
      }

      pcl::PointCloud<PointT>::Ptr seed_points(new pcl::PointCloud<PointT>());

      // 提取地面种子点
      extractInitialSeeds(pointcloud, seed_points);
      std::size_t num_points = pointcloud->size();

      Eigen::MatrixXf pc_eigen(num_points, FIELD_NUM);
      for (std::size_t j = 0; j < num_points; ++j) {
        pc_eigen(j, 0) = pointcloud->points[j].x;
        pc_eigen(j, 1) = pointcloud->points[j].y;
        pc_eigen(j, 2) = pointcloud->points[j].z;
      }

      Eigen::Array<bool, 1, -1> ground_mask;
      for (auto i = 0; i < iter_num_; i++) {

        // Convert pointcloud to Eigen matrix
        Eigen::MatrixXf ground_pc_eigen(seed_points->size(), FIELD_NUM);
        for (std::size_t j = 0; j < seed_points->size(); ++j) {
          ground_pc_eigen(j, 0) = seed_points->points[j].x;
          ground_pc_eigen(j, 1) = seed_points->points[j].y;
          ground_pc_eigen(j, 2) = seed_points->points[j].z;
        }

        // Perform PCA to get normal vector
        pcl::PCA<PointT> pca;
        pca.setInputCloud(seed_points);
        Eigen::Vector3f n = pca.getEigenVectors().col(2);
        n = n / n.norm();
        Eigen::Vector3f point_mean = pca.getMean().head(3);

        Eigen::VectorXf coeff(4);
        coeff.setZero();
        coeff.head(3) = n;
        coeff(3) = -point_mean.dot(n);

        Eigen::VectorXf dis = (pc_eigen * coeff.head(3)).array() + coeff(3);
        dis = dis.array().abs();

        ground_mask = dis.array() < inlier_threshold_;
        (*seed_points).clear();
        for (std::size_t j = 0; j < ground_mask.size(); j++) {
          if (ground_mask(j)) {
            seed_points->points.push_back(pointcloud->points[j]);
          }
        }
      }

      py::array_t<bool> output = py::array_t<bool>(num_points);
      py::buffer_info output_info = output.request();
      bool *output_ptr = (bool *) output_info.ptr;
      for (size_t i = 0; i < num_points; i++) {
        output_ptr[i] = ground_mask(i);
      }

      return output;
    }

    void extractInitialSeeds(const pcl::PointCloud<PointT>::Ptr &cloud_in,
                             const pcl::PointCloud<PointT>::Ptr &seed_points) {

      // 对激光点按高度进行排序
      std::vector<PointT> cloud_sorted((*cloud_in).points.begin(), (*cloud_in).points.end());
      sort(cloud_sorted.begin(), cloud_sorted.end(),
           [](PointT p1, PointT p2) {
               return p1.z < p2.z;
           }
      );

      // 移除因镜面反射而产生的噪点
      std::vector<PointT>::iterator it = cloud_sorted.begin();
      for (size_t i = 0; i < cloud_sorted.size(); ++i) {
        if (cloud_sorted[i].z < -1.5 * sensor_height_) {
          it++;
        } else {
          break;
        }
      }
      cloud_sorted.erase(cloud_sorted.begin(), it);

      // 找到最低的前num_lpr_个点，然后算其平均高度
      double LPR_height = 0.;
      for (int i = 0; i < num_lpr_; i++) {
        LPR_height += cloud_sorted[i].z;
      }
      LPR_height /= num_lpr_;

      // Iterate, filter for height less than LPR_height + th_seeds_
      (*seed_points).clear();
      for (size_t i = 0; i < cloud_sorted.size(); ++i) {
        if (cloud_sorted[i].z < LPR_height + seed_height_offset_) {
          (*seed_points).points.push_back(cloud_sorted[i]);
        }
      }
    }

private:
    float sensor_height_;
    float inlier_threshold_;
    int iter_num_;
    int num_lpr_;
    float seed_height_offset_;
};

PYBIND11_MODULE(gpf_pyb, m) {
  m.doc() = "an ground segmentation module for pointcloud";
  py::class_<GPF>(m, "GPF")
    .def(py::init<float, float, int, int, float>(),
         py::arg("sensor_height"),
         py::arg("inlier_threshold"),
         py::arg("iter_num"),
         py::arg("num_lpr"),
         py::arg("seed_height_offset"))
    .def("apply", &GPF::apply, "input"_a);
}
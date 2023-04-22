#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/common/angles.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

class cRangeImgCluster {
private:
    std::vector<std::vector<int>> neighbors = {{1,  0},
                                               {-1, 0},
                                               {0,  1},
                                               {0,  -1}};
    float horizon_res_;
    float vertical_res_;
    float threshold_h_;
    float threshold_v_;
    int valid_point_num_;
    int img_width_;
    int img_height_;
    float fov_top_;
    float fov_bottom_;
    float fov_left_;
    float fov_right_;

public:
    cRangeImgCluster(float horizon_res = 0.15, float vertical_res = 0.4,
                    float threshold_h = 10, float threshold_v = 10,
                    int valid_point_num = 30,
                    int img_width = 2400, int img_height = 64,
                    float fov_bottom = -24.9, float fov_top = 2,
                    float fov_left = 0, float fov_right = 360) {

      this->horizon_res_ = pcl::deg2rad(horizon_res);
      this->vertical_res_ = pcl::deg2rad(vertical_res);
      this->fov_top_ = fov_top;
      this->fov_bottom_ = fov_bottom;
      this->fov_left_ = fov_left;
      this->fov_right_ = fov_right;
      this->threshold_h_ = pcl::deg2rad(threshold_h);
      this->threshold_v_ = pcl::deg2rad(threshold_v);
      this->valid_point_num_ = valid_point_num;
      this->img_width_ = img_width;
      this->img_height_ = img_height;
    }

    std::vector<std::vector<int>> cluster(const py::array_t<float> &input) {

      cv::Mat range_img = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
      cv::Mat label_img = cv::Mat::zeros(range_img.size(), CV_32SC1);
      cv::Mat index_img = cv::Mat::ones(range_img.size(), CV_32SC1) * -1;

      auto input_ref = input.unchecked<2>();
      for (int i = 0; i < input_ref.shape(0); ++i) {
        float x = input_ref(i, 0);
        float y = input_ref(i, 1);
        float z = input_ref(i, 2);
        float range = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));

        // get angles
        float yaw = pcl::rad2deg(-std::atan2(y, x)) + 180;
        float pitch = pcl::rad2deg(std::asin(z / range));

        float img_x_resolution = img_width_ / (fov_right_ - fov_left_);
        float img_y_resolution = img_height_ / (fov_top_ - fov_bottom_);

        // round and clamp for use as index
        int img_x = std::floor((yaw - fov_left_) * img_x_resolution);
        int img_y = std::floor(img_height_ - (pitch - fov_bottom_) * img_y_resolution);

        img_x = std::clamp(img_x, 0, img_width_ - 1);
        img_y = std::clamp(img_y, 0, img_height_ - 1);

        if (index_img.at<float>(img_y, img_x) > 0) {
          if (range_img.at<float>(img_y, img_x) > range) {
            range_img.at<float>(img_y, img_x) = range;
            index_img.at<int>(img_y, img_x) = i;
          }
        } else {
          range_img.at<float>(img_y, img_x) = range;
          index_img.at<int>(img_y, img_x) = i;
        }
      }

      int label = 0;
      std::vector<std::vector<int>> cluster_idx_list;
      for (int i = 0; i < img_height_; i++) {
        for (int j = 0; j < img_width_; j++) {
          if (label_img.at<int>(i, j) == 0 && index_img.at<float>(i, j) >= 0) {
            label += 1;
            std::vector<int> cluster_idx = this->label_component_bfs(i, j, label, label_img, range_img, index_img);
             if (cluster_idx.size() > this->valid_point_num_) {
               cluster_idx_list.push_back(cluster_idx);
             }
          }
        }
      }

      return cluster_idx_list;
    }

    std::vector<int>
    label_component_bfs(int i, int j, int label, cv::Mat &label_img, cv::Mat &range_img, cv::Mat &index_img) {
      std::vector<int> cluster_idx = {index_img.at<int>(i, j)};
      std::queue<std::vector<int>> queue;
      queue.push(std::vector<int>({i, j}));

      while (!queue.empty()) {
        std::vector<int> target = queue.front();
        queue.pop();

        int target_i = target[0];
        int target_j = target[1];

        if (label_img.at<int>(target_i, target_j) == 0) {
          label_img.at<int>(target_i, target_j) = label;
        }

        for (auto &neighbor: this->neighbors) {
          int r = neighbor[0];
          int c = neighbor[1];
          int neighbor_i = target_i + r;
          int neighbor_j = target_j + c;

          // margin case
          if (neighbor_i < 0 || neighbor_i >= this->img_height_) {
            continue;
          }
          if (neighbor_j < 0) {
            neighbor_j = this->img_width_ - 1;
          }
          if (neighbor_j >= this->img_width_) {
            neighbor_j = 0;
          }
          if (label_img.at<int>(neighbor_i, neighbor_j) != 0) {
            continue;
          }
          if (index_img.at<int>(neighbor_i, neighbor_j) < 0) {
            continue;
          }

          double alpha = this->horizon_res_;
          double threshold_ = this->threshold_h_;
          if (c == 0) {
            alpha = this->vertical_res_;
            threshold_ = this->threshold_v_;
          }

          double range_neighbor = range_img.at<float>(neighbor_i, neighbor_j);
          double range_target = range_img.at<float>(target_i, target_j);

          double d1 = std::max(range_neighbor, range_target);
          double d2 = std::min(range_neighbor, range_target);

          double threshold = std::atan2((d2 * std::sin(alpha)), (d1 - d2 * std::cos(alpha)));

          if (threshold > threshold_) {
            label_img.at<int>(neighbor_i, neighbor_j) = label;
            cluster_idx.push_back(index_img.at<int>(neighbor_i, neighbor_j));
            queue.push({neighbor_i, neighbor_j});
          }
        }
      }

      return cluster_idx;
    }
};

PYBIND11_MODULE(range_img_cluster_pyb, m) {
  m.doc() = "an cluster segmentation module for pointcloud";
  py::class_<cRangeImgCluster>(m, "cRangeImgCluster")
    .def(py::init<float, float, float, float, int, int, int, float, float, float, float>(),
         py::arg("horizon_res") = 0.15f,
         py::arg("vertical_res") = 0.4f,
         py::arg("threshold_h") = 10.0f,
         py::arg("threshold_v") = 10.0f,
         py::arg("valid_point_num") = 30,
         py::arg("img_width") = 2400,
         py::arg("img_height") = 64,
         py::arg("fov_bottom") = -24.9f,
         py::arg("fov_top") = 2.0f,
         py::arg("fov_left") = 0.0f,
         py::arg("fov_right") = 360.0f
    )
    .def("cluster", &cRangeImgCluster::cluster, "input"_a);
}
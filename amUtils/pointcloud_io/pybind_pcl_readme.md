 

 

# 封装pcd文件的读取

重点是构建一个numpy的c++封装类 `py::array_t<float>`来作为c++和python点云数据互通的桥梁

## 代码

步骤一：创建文件 `load_pcd_file_pcl.cpp` 

```c++
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

// 生成c++拓展库
py::array_t<float> load_pcd_file(std::string &file_name, bool remove_nan) {
    pcl::PointCloud<PointType>::Ptr pointcloud(new pcl::PointCloud<PointType>);
    std::cout << "loading file: " << file_name << std::endl;
    if (pcl::io::loadPCDFile<PointType>(file_name, *pointcloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    if (remove_nan) {
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*pointcloud, *pointcloud, indices);
    }

    unsigned int pointcloud_size = pointcloud->width * pointcloud->height;
    py::array_t<float, py::array::c_style> pointcloud_output({(const int) pointcloud_size, FIELD_NUM});

    // 用这种方法可以使用()的赋值方法
    // 2表示dimensions
    auto pointcloud_proxy = pointcloud_output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < pointcloud_proxy.shape(0); i++) {
        pointcloud_proxy(i, 0) = pointcloud->points[i].x;
        pointcloud_proxy(i, 1) = pointcloud->points[i].y;
        pointcloud_proxy(i, 2) = pointcloud->points[i].z;
        pointcloud_proxy(i, 3) = pointcloud->points[i].intensity;
    }

    return pointcloud_output;
}

PYBIND11_MODULE(load_pcd_file, m) {
    m.doc() = "load pcd pointcloud and achieve np-format pointcloud";
    m.def("load_pcd_file", &load_pcd_file, "load pcd file",
          "file_name"_a, "remove_nan"_a = true);
};
```



## 解读代码

创建`py::array`数组的方法：

方法一：

```c++
// 方法一：
auto pointcloud_output = py::array_t<float, py::array::c_style>(
    py::array::ShapeContainer({pointcloud_size, FIELD_NUM})  );
```

方法二：

```c++
// 第二个模板参数指的是storage layout是row-major(c_style)还是column-major(f_style)
py::array_t<float, py::array::c_style> pointcloud_output({(const int) pointcloud_size, FIELD_NUM});
```

.. note:: 该方法需要将类型转换为只读（加上const）否则会报错 "No matching constructor for initialization of 'py::array_t<float, py::array::c_style>'"

## 编译和运行代码

步骤一：创建CMakeLists.txt文件

```cmake
cmake_minimum_required(VERSION 3.10)
project(PcdIo)

find_package(PCL 1.11 QUIET)
find_package(pybind11)

include_directories(${PCL_INCLUDE_DIRS})
pybind11_add_module(load_pcd_file_pcl load_pcd_file_pcl.cpp)
target_link_libraries(load_pcd_file_pcl ${PCL_LIBRARIES})
```

PS: 

- [不显示pcl warning](https://github.com/PointCloudLibrary/pcl/issues/3680)

步骤二：从python调用封装函数

```c++
$ python
>>> import load_pcd_file_pcl
>>> load_pcd_file_pcl.load_pcd_file("<...>.pcd",False)
array([[ 2.4369328,  7.4470015,  2.09954  ,  0.       ],
       [ 2.450845 ,  7.3976755,  2.0881522,  0.       ],
       [ 2.4717412,  7.3744216,  2.0840108,  0.       ],
       ...,
       [ 2.1694767, -2.066687 , -0.8028567,  1.       ],
       [ 2.1427321, -2.0555243, -0.7956098,  1.       ],
       [ 2.1661112, -2.09252  , -0.8069978,  1.       ]], dtype=float32)
>>> 
```


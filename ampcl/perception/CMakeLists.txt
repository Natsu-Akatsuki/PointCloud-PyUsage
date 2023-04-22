cmake_minimum_required(VERSION 3.11)
project(filter)

find_package(PCL QUIET)
find_package(OpenCV REQUIRED)

find_package(pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PCL_INCLUDE_DIRS})
pybind11_add_module(gpf_pyb src/gpf_pyb.cpp)
target_link_libraries(gpf_pyb PRIVATE ${PCL_LIBRARIES})

pybind11_add_module(euclidean_cluster_pyb src/euclidean_cluster_pyb.cpp)
target_link_libraries(euclidean_cluster_pyb PRIVATE ${PCL_LIBRARIES})

include_directories(${OpenCV_INCLUDE_DIRS})
pybind11_add_module(range_img_cluster_pyb src/range_img_cluster_pyb.cpp)
target_link_libraries(range_img_cluster_pyb PRIVATE ${PCL_LIBRARIES} ${OpenCV_LIBS})
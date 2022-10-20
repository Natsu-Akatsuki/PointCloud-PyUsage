cmake_minimum_required(VERSION 3.11)
project(pointcloud_io)

find_package(PCL QUIET)
find_package(pybind11)
set(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PCL_INCLUDE_DIRS})
message(CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR})
pybind11_add_module(io_pyb src/io_pyb.cpp)

target_link_libraries(io_pyb PRIVATE ${PCL_LIBRARIES})

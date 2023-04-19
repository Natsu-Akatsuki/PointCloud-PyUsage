cmake_minimum_required(VERSION 3.11)
project(filter)

find_package(PCL QUIET)
find_package(pybind11)
set(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PCL_INCLUDE_DIRS})
pybind11_add_module(gpf_pyb src/gpf_pyb.cpp)

target_link_libraries(gpf_pyb PRIVATE ${PCL_LIBRARIES})

cmake_minimum_required(VERSION 3.10)
project(pointcloud_io)

find_package(PCL QUIET)
find_package(pybind11)
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(${PCL_INCLUDE_DIRS})
pybind11_add_module(load_pcd_file_pcl load_pcd_file_pcl.cpp)

target_link_libraries(load_pcd_file_pcl PRIVATE ${PCL_LIBRARIES})
set_target_properties(load_pcd_file_pcl PROPERTIES PREFIX "")                         # 指定前缀
set_target_properties(load_pcd_file_pcl PROPERTIES OUTPUT_NAME "load_pcd_file_pcl")   # 指定文件名
set_target_properties(load_pcd_file_pcl PROPERTIES SUFFIX ".so")                      # 指定后缀
set_target_properties(load_pcd_file_pcl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定库的输出路径


pybind11_add_module(save_pcd_file_pcl save_pcd_file_pcl.cpp)
target_link_libraries(save_pcd_file_pcl PRIVATE ${PCL_LIBRARIES})
set_target_properties(save_pcd_file_pcl PROPERTIES PREFIX "")                         # 指定前缀
set_target_properties(save_pcd_file_pcl PROPERTIES OUTPUT_NAME "save_pcd_file_pcl")   # 指定文件名
set_target_properties(save_pcd_file_pcl PROPERTIES SUFFIX ".so")                      # 指定后缀
set_target_properties(save_pcd_file_pcl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})  # 指定库的输出路径
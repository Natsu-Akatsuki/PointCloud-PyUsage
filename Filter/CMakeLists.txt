cmake_minimum_required(VERSION 3.3)
set(CMAKE_BUILD_TYPE "RELEASE")
add_compile_options("-O3")
project(pointcloud_filter)

set(CMAKE_BUILD_TYPE "RELEASE")
add_compile_options("-O3")

find_package(
  catkin REQUIRED
  COMPONENTS geometry_msgs
             sensor_msgs
             roscpp
             rospy
             std_msgs
             pcl_ros
             tf)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  DEPENDS
  EIGEN3
  PCL
  INCLUDE_DIRS)

add_executable(pointcloud_filter pointcloud_filter.cpp)
target_link_libraries(pointcloud_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES})

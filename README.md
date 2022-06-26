# PointCloud-PyUsage

提供点云处理的python实现

## Requirement

- 测试于ubuntu20.04

```bash
$ cd PointCloud-PyUsage
# install
$ pip3 install -e .
# 使用pybind后的拓宽库
$ python3 setup.py build_cmake_ext
# uninstall
$ python3 setup_cpu.py uninstall
# clean
$ python3 setup_cpu.py clean
```

## Outline(TODO)

| file/package                              | description                                                  |
| ----------------------------------------- | ------------------------------------------------------------ |
| io_and_transform.cpp                      | pcl io and data type transform<br />pcl<->vector; pcd<->pcl; bin->pcl<br /> |
| restore_pointcloud_ring_info              |                                                              |
| ros_numpy                                 | numpy array <-> ros msg                                      |
| ddynamic_reconfigure_python               |                                                              |
| pointcloud_vis/intensity_to_color.py      | get pseudo color based on intensity                          |
| livox_msg_tran.py                         | Livox customMsg-> numpy (**note**: it costs 50ms), require dependency [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) |
| visualize/3dod/example.py                 | visualize pointcloud and 3D bbx based on **open3d**<br />remove foreground points |
| visualize/3dod/visualize_livox_dataset.py | visualize livox_dataset colored pointcloud on **rviz**       |
| visualize/3dod/visualize_pcdet_result.py  | visualize OpenPCDet result on **rviz**                       |
| visualize/stereo/dis2pointcloud.py        | tranform disparity image to pointcloud                       |

## Usage(TODO)

- dis2pointcloud

<p align="center">
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220413194957408.png" alt="image-20220413194957408" width=50% height=50% />
</p>

```bash
$ cd visualize/stereo/
$ python3 dis2pointcloud.py
```

- visualize_pcdet_result

```bash
# step1: install dependency
$ apt install ros-noetic-jsk-recognition-msgs ros-noetic-jsk-rviz-plugins
# step2: now the variable is hard coding, you should modify the path in the visualize_pcdet_result.py
# step3
$ cd visualize/3dod
$ python3 visualize_pcdet_result.py
# step4: open rviz
$ rviz -d kitti.rviz
```

<p align="center">
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220412135243609.png" alt="image-20220412135243609.png" width=50% height=50% />
</p>

## Log

1. 2022.06.27 v0.1.2 参考[PEP 621](https://peps.python.org/pep-0621/)规范打包python模块

## Reference

- [ddynamic_reconfigure_python](https://github.com/pal-robotics/ddynamic_reconfigure_python)
- [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver)
- [livox_mapping](https://github.com/Livox-SDK/livox_mapping)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [pykitti](https://github.com/utiasSTARS/pykitti)
- [ros_numpy](https://github.com/eric-wieser/ros_numpy)

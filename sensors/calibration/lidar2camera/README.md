# ROS手动标定工具

![image-20221125201007016](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20221125201007016.png)

上：从左到右分别是原图、强度、深度图（深度：三维点到相机光心的欧式距离）
下：彩色点云图

## Requirement

- `ampcl`，具体安装方式看根目录`README`

## Usage

- 启动上色和投影

```bash
(ros1) $ python3 manual_calibration_ros1.py
(ros2) $ python3 manual_calibration_ros2.py
```

- 动态调参

```bash
(ros1) $ rosrun rqt_reconfigure rqt_reconfigure
(ros2) $ ros2 run rqt_reconfigure rqt_reconfigure
```

- `RViz`可视化

```bash
# note: 点云的坐标系为lidar
(ros1) $ rviz -d rviz1.rviz
(ros2) $ rviz -d rviz2.rviz
```

## Mode

- 目前代码为hard-coded，具体在`__main__`中进行修改：

|  模式  |                             作用                             |
| :----: | :----------------------------------------------------------: |
| mode 0 |            读取点云文件和图片，用Open3D进行可视化            |
| mode 1 |  读取点云文件和图片，用RVIZ进行可视化，可动态调参，定时发布  |
| mode 2 | 读取ROS点云数据和图片，用RVIZ进行可视化，可动态调参，定时发布 |
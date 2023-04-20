## README

含地面分割、聚类分割、形状拟合等算法的全家桶示例

### Non-ROS Version

<img src="docs/traditional_perception.png" alt="image-20230420165640384" style="zoom:67%;" />

```bash
$ python3 traditional_perception.py --pc_dir <点云文件夹>
```

### ROS Version

>**Note**
>
>暂时只支持KITTI数据集

- 步骤一：修改`config/kitti.yaml`中的相关参数，如数据集路径
- 步骤二：

```bash
$ python3 ros2_traditional_perception_kitti.py 
```

- 步骤三：启动可视化界面

```bash
$ rviz2 -d kitti_ros2.rviz 
```

<img src="docs/ros2_traditional_perception_kitti.png" alt="image-20230420193043312" style="zoom:80%;" />

## TODO

- [ ] 追加其他数据集的支持
- [ ] 追加对ROS1的支持
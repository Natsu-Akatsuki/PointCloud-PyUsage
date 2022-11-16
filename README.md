# AmPCL

提供点云处理的`Python`实现

## 安装

### 安装依赖

- ROS1 / ROS2（如需要用到ROS的相关插件）
- 构建工具

```bash
$ pip3 install -U --user build pip setuptools wheel
```

### 构建安装和安装包


```bash
$ git clone https://github.com/Natsu-Akatsuki/PointCloud-PyUsage --depth=1
$ cd PointCloud-PyUsage
$ bash install.sh
```

> **Note**
>
> 使用开发模式可能出现`its build backend is missing the 'build_editable' hook`的报错，则可能是系统级别的`setuptools`的版本覆盖了高版本的`setuptools`。通过如下命令行可查询当前的版本`python3 -c "import setuptools; print(setuptools.__version__)" `

## 程序

|      包名       |                    作用                    |
| :-------------: | :----------------------------------------: |
|      `io`       |             导入和导出点云文件             |
| `visualization` |                 点云可视化                 |
|   `ros-utils`   | 动态调参，ROS消息类型和numpy类型的相互转换 |
|    `filter`     |            点云下采样和直通滤波            |

### IO

- 支持`npy`，`pcd`，`bin`点云文件的读取
- 目前支持读取的字段包括`xyz`，`xyzi`

```python
# >>> import usage >>>
from ampcl.io import load_pointcloud

# low-level API
pointcloud = load_npy("ampcl/data/pointcloud.npy")
pointcloud = load_txt("ampcl/data/pointcloud.txt")
pointcloud = load_pcd("ampcl/data/pointcloud_ascii.pcd")
# high-level API（支持npy, pcd, bin文件）
pointcloud = load_pointcloud(".ampcl/data/pointcloud.pcd")

# >>> export usage >>>
save_pointcloud(pointcloud, "pointcloud.pcd")

from pointcloud_utils.io import c_load_pcd
c_load_pcd(".ampcl/data/pointcloud_ascii.pcd")
```

### Visualization

- 可视化激光点云
- 基于强度实现点云伪彩色增强

```python
from ampcl.io import load_pointcloud
from ampcl.visualization import o3d_viewer_from_pointcloud
pointcloud = load_pointcloud("ampcl/data/pointcloud_ascii.pcd")
o3d_viewer_from_pointcloud(pointcloud)
```

## 命令行

### 可视化点云

- 基于`Open3D`的点云可视化，支持`npy`，`pcd`，`bin`点云文件的可视化，目前支持的字段包括`xyz`，`xyzi`
- 强度字段使用伪彩色增强

```bash
$ o3d_viewer <pointcloud_file>
# 如遇到KITTI数据集这种将强度进行过归一化的则需要加上-n选项
$ o3d_viewer -n <pointcloud_file>
```

<p align="center">
<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20221020191241065.png" alt="img" width=80% height=80% />
</p>

## BUG

- [ ] 不论是否处于虚拟环境，用开发模式安装的包都无法被`Pycharm`智能识别，`Cannot find reference`
- [ ] `build`构建虚拟环境后，执行的`pip`为非用户路径下的`pip`，如若系统路径未安装`pip`，则会显示`No module named pip`

## 规范

- 遵从`pep660`（`setuptools`至少需要`v64.0`才支持单个toml下的`develop`模式）（@[ref](https://stackoverflow.com/questions/69711606/how-to-install-a-package-using-pip-in-editable-mode-with-pyproject-toml)）

## 参考资料

|                             仓库                             |                 参考                 |
| :----------------------------------------------------------: | :----------------------------------: |
|    [ros_numpy](https://github.com/eric-wieser/ros_numpy)     | `ROS1` 点云数据和`numpy`数据相互转换 |
|   [ros2_numpy](https://github.com/Box-Robotics/ros2_numpy)   | `ROS1` 点云数据和`numpy`数据相互转换 |
| [ddynamic_reconfigure_python](https://github.com/pal-robotics/ddynamic_reconfigure_python) |           `ROS1` 动态调参            |
|     [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)     |          神经网络算子和标定          |
|       [pykitti](https://github.com/utiasSTARS/pykitti)       |          `KITTI`数据集读取           |
| [livox_mapping](https://github.com/Livox-SDK/livox_mapping)  |          基于强度值的伪彩色          |

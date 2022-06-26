# 用例

- ros_subscribe_usage

订阅ros点云并转换为numpy类型（点云类型默认为xyzi）

```python
# ros pointcloud -> np pointcloud
import pointcloud_io
pointcloud_io.ros_subscribe_usage(topic_name="/rslidar_points")
```

- load_bin

加载bin格式的点云文件

```python
load_bin("../data/000000.bin")
```

- load_npy和save_pointcloud

读取npy格式的点云文件，同时基于封装的pcl点云库的API来保存点云文件

```python
import pointcloud_io
import numpy as np

pointcloud = pointcloud_io.load_npy("data/000000.npy")[:, :4]
pointcloud_io.save_pointcloud(pointcloud_np=pointcloud, file_name="pointcloud.pcd", method="pcl")
```


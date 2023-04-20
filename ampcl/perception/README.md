# Usage

## Ground segmentation

### Implementation

#### RANSAC

- 论文：Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography
- 示例代码：

```python
from ampcl.perception import ground_segmentation_ransac
from ampcl.io import load_pointcloud

pc_np = load_pointcloud("...")
# 设置地平面样本点的大概位置 (x_min, x_max, y_min, y_max, z_min, z_max)
limit_range = (0, 50, -10, 10, -2.5, -0.5)
plane_model, ground_mask = ground_segmentation_ransac(pc_np, limit_range, distance_threshold=0.2, debug=True)
ground_pc = pc_np[ground_mask]
non_ground_pc = pc_np[~ground_mask]
```

![image-20230321230330622](docs/ransac-indoor.png)

#### GPF

- **论文**：Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications
- **亮点**：coarse to fine（更准确的地面点和更准确的平面模型）, PCA计算地平面模型
- **算法流程**：

1. 去除因镜面反射而产生的噪点（移除太低的点：-1.5×传感器相对地面的高度）
2. 选取地面种子点（前n个z值最低的点的均值+高度offset作为种子点的z轴阈值，保留低于该阈值的点），基于PCA生成地平面模型
3. 基于地平面模型，获取新的地面点，并基于该新的地面点更新地平面模型，重复该步骤多次

- **示例代码**：

```python
# pybind版本（recommend）
from ampcl.perception import ground_segmentation_gpf
from ampcl.io import load_pointcloud

pc_np = load_pointcloud("...")
ground_mask = ground_segmentation_gpf(pc_np, debug=True)


# python版本
from ampcl.perception.ground_segmentation import GPF
from ampcl.io import load_pointcloud

pc_np = load_pointcloud("...")
gpf = GPF()
non_ground_pointcloud = gpf.apply(pc_np, debug=False)
```

<img src="docs/gpf-kitti.png" alt="image-20230420013022459" style="zoom:67%;" />



### Data

#### Speed

|       Method        | Speed |
| :-----------------: | :---: |
|       LineFit       |   —   |
|         GPF         |   —   |
|       RANSAC        |   —   |
|        R-GPF        |   —   |
|     CascadedSeg     |   —   |
|      Patchwork      |   —   |
|     Patchwork++     |   —   |
| Patchwork++ w/o TGR |   —   |

## TODO

- [ ] 添加其他方案的Python实现和C++实现
- [ ] 感受一波[Ground Segmentation BenchMark](https://github.com/url-kaist/Ground-Segmentation-Benchmark)

## Supplementation

- [awesome ground segmentation](docs/awesome_ground_segmentation.md)

## 

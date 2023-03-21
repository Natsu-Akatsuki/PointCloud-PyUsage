# Usage

## Ground segmentation

### Model-based

#### PCL

```python
from ampcl.perception import ground_segmentation_ransac
from ampcl.io import load_pointcloud

pc = load_pointcloud("...")
# 设置地平面的大概位置 (x_min, x_max, y_min, y_max, z_min, z_max)
limit_range = (0, 50, -10, 10, -2.5, -0.5)
coeff, ground_pc_idx = ground_segmentation_ransac(pc, limit_range, distance_threshold=0.2, debug=True)
ground_pc = pc[ground_pc_idx]
non_ground_pc = pc[~ground_pc_idx]
```

![image-20230321230330622](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20230321230330622.png)

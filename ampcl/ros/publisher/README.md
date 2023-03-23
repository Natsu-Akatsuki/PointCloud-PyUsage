```
# ...
from ampcl.ros import publisher

latching_qos = QoSProfile(depth=5, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
self.pc_in_pub = self.create_publisher(PointCloud2, "/debug/pointcloud/in_region", latching_qos)
self.pc_out_pub = self.create_publisher(PointCloud2, "/debug/pointcloud/out_region", latching_qos)
self.limit_range = (0, -44.8, -2.0, 99.6, 44.8, 2.0)

publisher.publish_pc_by_range(self.pc_in_pub, self.pc_out_pub, pointcloud,
                              header,
                              (0, -44.8, -2.0, 99.6, 44.8, 2.0),
                              field="xyzirgb")
```

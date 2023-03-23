```python
from ampcl.ros import marker

# incomplete usage
latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
distance_marker = marker.create_distance_marker(frame_id="...", distance_delta=10)
distance_marker_pub = self.create_publisher(MarkerArray, "/debug/distance_marker", latching_qos)
distance_marker_pub.publish(distance_marker)
```
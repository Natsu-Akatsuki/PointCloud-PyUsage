```python
from ampcl.ros import marker

# incomplete usage
latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
distance_marker = marker.create_distance_marker(frame_id="...", distance_delta=10)
distance_marker_pub = self.create_publisher(MarkerArray, "/debug/distance_marker", latching_qos)
distance_marker_pub.publish(distance_marker)
```

```python
from ampcl.ros import marker

box3d_marker_array = MarkerArray()
empty_marker = Marker()
empty_marker.action = Marker.DELETEALL
box3d_marker_array.markers.append(empty_marker)

box3d_marker_array, _ = marker.create_gt_detection_box3d_marker_array(pointcloud, gt_box3d_lidar,
                                                                      pc_color=None,
                                                                      stamp=stamp, frame_id=self.frame_id,
                                                                      box3d_marker_array=box3d_marker_array,
                                                                      box3d_ns="gt/detection/box3d"
                                                                      )

pred_box3d_lidar = pred_infos['box3d_lidar']
score = pred_infos['score']
box3d_marker_array, _ = marker.create_pred_detection_box3d_marker_array(pointcloud, pred_box3d_lidar,
                                                                        score=score,
                                                                        pc_color=None,
                                                                        stamp=stamp,
                                                                        frame_id=self.frame_id,
                                                                        box3d_arr_marker=box3d_marker_array,
                                                                        box3d_ns="pred/detection/box3d"
                                                                        )

box3d_marker_array, colors_np = marker.create_pred_tracker_box3d_marker_array(pointcloud, trackers,
                                                                              pc_color=colors_np,
                                                                              stamp=stamp,
                                                                              frame_id=self.frame_id,
                                                                              box3d_arr_marker=box3d_marker_array,
                                                                              box3d_ns="pred/tracker/box3d",
                                                                              tracker_text_ns="pred/tracker/text/tracker_id"
                                                                              )

```


```python
# 创建真值框
# 其中真值框的颜色与类别有关（默认情况下颜色为黑色）
# 对真值框中的点云进行着色
def create_gt_box3d_marker_array(self, box3d_marker_array, box3d_lidar, stamp, pc_np=None, pc_color=None):
    box3d_color = []
    for i in range(box3d_lidar.shape[0]):
        a_box3d_lidar = box3d_lidar[i]
        cls_id = int(a_box3d_lidar[7])
        box3d_color.append(kitti_cls_id_to_color(cls_id))

        if pc_np is not None and pc_color is not None:
            inside_points_idx = get_indices_of_points_inside(pc_np, a_box3d_lidar, margin=0.1)
            pc_color[inside_points_idx] = box3d_color

    marker.create_box3d_marker_array(box3d_lidar, box3d_color=box3d_color,
                                     stamp=stamp, frame_id=self.frame_id,
                                     box3d_marker_array=box3d_marker_array,
                                     box3d_ns="gt/detection/box3d",
                                     line_width=0.2
                                     )
```



```python
# for debug: 发布激光雷达二维八角点预测框
img_pub = img.copy() if pred_box2d8c_lidar.shape[0] == 0 \
    else paint_box3d_on_img(img.copy(), pred_box2d8c_lidar, cls_idx=pred_box3d_lidar[:, 7])
publisher.publish_img(img_pub, self.pub_dict["/pred/img/box2d8c_lidar"], stamp, self.frame_id)

# for debug: 发布激光雷达二维四角点预测框
img_pub = img.copy() if pred_box2d8c_lidar.shape[0] == 0 \
    else paint_box2d_on_img(img.copy(), pred_box2d4c_lidar, cls_idx=pred_box3d_lidar[:, 7])
publisher.publish_img(img_pub, self.pub_dict["/pred/img/box2d4c_lidar1"], stamp, self.frame_id)
```
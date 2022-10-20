try:
    import rclpy

    from .ros2_numpy.point_cloud2 import *
except:
    try:
        import rospy

        from .ddynamic_reconfigure_python.ddynamic_reconfigure import \
            DDynamicReconfigure
        from .ros_numpy.point_cloud2 import *
    except:
        raise ImportError("Please install ROS2 or ROS1")


def pointcloud2_to_np(pointcloud2, remove_nans=True, field="xyzi"):
    if field == "xyz":
        return pointcloud2_to_xyz_array(pointcloud2, remove_nans=remove_nans)
    elif field == "xyzi":
        return pointcloud2_to_xyzi_array(pointcloud2, remove_nans=remove_nans)
    else:
        raise NotImplementedError("field: {} isn't implemented!".format(field))


def np_to_pointcloud2(pointcloud_np, field="xyzi"):
    if field == "xyz":
        return xyz_numpy_to_pointcloud2(pointcloud_np)
    elif field == "xyzi":
        return xyzi_numpy_to_pointcloud2(pointcloud_np)
    elif field == "xyzrgb":
        return xyzrgb_numpy_to_pointcloud2(pointcloud_np)
    elif field == "xyzirgb":
        return xyzirgb_numpy_to_pointcloud2(pointcloud_np)
    else:
        raise NotImplementedError("field {} isn't implemented!".format(field))

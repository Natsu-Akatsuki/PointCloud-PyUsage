from .. import np_to_pointcloud2
from ...filter import passthrough_filter

def publish_pc_by_range(in_pub, out_pub, pointcloud, header, limit_range,
                        point_wise_height_offset=0.0,
                        field="xyzirgb"):
    """
    根据直通滤波的范围对点云进行分组，并发布
    :param point_wise_height_offset:
    :param in_pub:
    :param out_pub:
    :param pointcloud:
    :param header:
    :param limit_range:
    :param field:
    :return:
    """
    mask = passthrough_filter(pointcloud[:, :3], list(limit_range))

    pointcloud[:, 2] += point_wise_height_offset

    pointcloud_in = pointcloud[mask]
    pointcloud_out = pointcloud[~mask]

    if in_pub.get_subscription_count() > 0:
        pointcloud_msg = np_to_pointcloud2(pointcloud_in, header, field=field)
        in_pub.publish(pointcloud_msg)

    if out_pub.get_subscription_count() > 0:
        pointcloud_msg = np_to_pointcloud2(pointcloud_out, header, field=field)
        out_pub.publish(pointcloud_msg)

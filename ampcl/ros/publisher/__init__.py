from .pc_publisher import *
from .img_publisher import *

import std_msgs.msg


def create_header(stamp, frame_id):
    header = std_msgs.msg.Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

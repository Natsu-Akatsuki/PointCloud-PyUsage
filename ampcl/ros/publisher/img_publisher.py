from cv_bridge import CvBridge

bridge = CvBridge()


def publish_img(img, img_pub, stamp, frame_id):
    if img_pub.get_subscription_count() > 0:
        img_msg = bridge.cv2_to_imgmsg(cvim=img, encoding="passthrough")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame_id
        img_pub.publish(img_msg)

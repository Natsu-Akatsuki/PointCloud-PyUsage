import glob
import os

import cv2
import rosbag
import rospy
from cv_bridge import CvBridge
from tqdm import tqdm


class Img2Bag:
    def __init__(self):
        self.output_bag = "./output.bag"
        self.input_dir = "/home/helios/data/livox_img/"
        self.freq = 30

        # self.r = rospy.Rate(self.freq)
        self.bridge = CvBridge()
        exts = ['png', 'jpg']
        self.img_list = []
        for ext in exts:
            self.img_list.extend(glob.glob(self.input_dir + '*.' + ext))

        self.img_list = sorted(self.img_list)

    def run(self):
        bag = rosbag.Bag(self.output_bag, 'w')
        r = rospy.Rate(self.freq)
        for img_file in tqdm(self.img_list):
            if rospy.is_shutdown():
                rospy.logrror("rospy is_shutdown, the transfer is interrupted")
                break
            image = cv2.imread(img_file)
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            image_msg.header.frame_id = "camera"
            bag.write("/camera/image_raw", image_msg, rospy.Time.now())
            r.sleep()

        bag.close()


if __name__ == '__main__':
    rospy.init_node('img2rosbag', anonymous=False)
    img2bag = Img2Bag()
    img2bag.run()
    rospy.loginfo("export rosbag " + img2bag.output_bag + " successfully")

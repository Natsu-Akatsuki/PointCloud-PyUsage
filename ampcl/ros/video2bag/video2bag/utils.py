from __future__ import print_function

import re
import time
from glob import glob
from os import makedirs, path

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class V2BConverter:
    def __init__(self, input_path, output_file, **kwargs):
        self.bag = None
        self.input_path = input_path
        self.output_file = output_file
        self.output_dir = kwargs.get('output_dir', './')
        self.sleep_rate = kwargs.get('sleep_rate', 0.1)
        self.div_num = kwargs.get('div_num', 2)

    @staticmethod
    def open_output_dir(output_dir):
        try:
            makedirs(output_dir)
            print("Directory ", output_dir, " Created")
        except FileExistsError:
            print("Directory ", output_dir, " already exists")

    def open_bag_file(self):
        try:
            self.bag = rosbag.Bag(path.join(self.output_dir, self.output_file), 'w')
        except Exception as e:
            print(e)

    def write_image(self, image):
        bridge = CvBridge()

        try:
            image_message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.bag.write('/camera/image',  image_message)
            time.sleep(self.sleep_rate)
        except Exception as e:
            print(e)

    def convert(self):
        cap = cv2.VideoCapture(self.input_path)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, True)

        self.open_output_dir(self.output_dir)
        self.open_bag_file()

        i, count = 0, 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret is False:
                break
            if i % self.div_num == 0:
                # im_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Used to remove the time written over the image frame
                # im_crop = im_bgr[:980, :1919]
                # Resize resolution
                # im_resize = cv2.resize(im_bgr, None, fx=0.5, fy=0.5)
                # cv2.imwrite(output_dir + 'extracted_frame_' + str(count) + '.jpg', im_resize)
                self.write_image(frame)
                print("Wrote extracted_frame_"+str(count)+'.jpg')
                count += 1
            i += 1

        print("Total {} of frames are made".format(count))
        cap.release()
        cv2.destroyAllWindows()

        # 一定要加close，不然rosbag文件是不完整的。后续打开时需要rosbag index来修复
        self.bag.close()

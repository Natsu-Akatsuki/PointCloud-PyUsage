from __future__ import print_function

from utils import V2BConverter


def parser():
    import argparse
    basic_desc = "convert video file(.mp4) to rosbag(.bag) file"
    main_parser = argparse.ArgumentParser(descriptio=basic_desc, add_help=False)
    options = main_parser.add_argument_group("convert options")

    options.add_argument("input_file",
                         help="path to a video file want to convert")
    options.add_argument("--output_file",
                         help="name of output bag file")
    options.add_argument("--output_dir",
                         help="directory of output bag file")
    options.add_argument("--sleep_rate",
                         help="time interval between video frames")
    options.add_argument("--div_num",
                         help="skip cycle of video frames")

    return main_parser


def run(args):
    import os
    import sys

    converter = V2BConverter(args.input_file, args.output_file, vars(args))
    converter.convert()


if __name__ == "__main__":
    input_file = "./img_test01_camera_raw_img.mp4"
    output_file = "./output.bag"
    converter = V2BConverter(input_file, output_file, output_dir="./", sleep_rate=0.1, div_num=2)
    converter.convert()

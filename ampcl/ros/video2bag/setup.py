from setuptools import setup

setup(
    name='video2bag',
    version='1.0.0',
    description='package for processing video file(.mp4) to rosbag(.bag) file',
    url='https://github.com/Cecilimon/video2bag',
    packages=['video2bag'],
    install_requires=['glob', 'regex'],
    classifiers=['Programming Language :: Python :: 2.7']
)

# video2bag
package for processing video file(.mp4) to **rosbag(.bag)** file

## prerequisites
- opencv-python
- cv_bridge
- rosbag
- glob
- regex

## Installation
**pip** : glob, regex <br>
**apt** : opencv-python, ros-{distro}-cv-bridge, ros-{distro}-desktop-full

## Instructions
### 1. indicate the directories of your input video and output frame in the main.py file
```
input_file = "./test.mp4"
output_file = "./output.bag"
args = {"output_dir": "./", "sleep_rate": 0.1, "div_num": 2}
```

### 2. Extract frame and run video2bag pkg by using main.py

```
python main.py
```

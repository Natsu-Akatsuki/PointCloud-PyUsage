import numpy as np
import os
import argparse
from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='KITTI label data analysis')
    parser.add_argument('--label_dir', type=str, default='/path/to/label/dir', help='path to label directory')
    parser.add_argument('--stat_dir', type=str, default='kitti_stats', help='path to statistics directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    label_dir = args.label_dir
    label_files = os.listdir(label_dir)
    stat_dir = Path(args.stat_dir).resolve()
    stat_dir.mkdir(exist_ok=True)

    stats = defaultdict(list)
    for label_file in tqdm(label_files):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.strip().split(' ')
                cls_type = label[0]

                if cls_type not in ['Car', 'Pedestrian', 'Cyclist']:
                    continue

                truncation = float(label[1])
                occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
                alpha = float(label[3])
                box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
                h = float(label[8])
                w = float(label[9])
                l = float(label[10])
                if h > 9:
                    print(f"height={h}, file={label_file}, line={line}")
                loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
                ry = float(label[14])
                score = float(label[15]) if label.__len__() == 16 else -1.0

                stats[f'{cls_type}_length'].append(l)
                stats[f'{cls_type}_width'].append(w)
                stats[f'{cls_type}_height'].append(h)

    for key in stats.keys():
        print(
            f'{key}: mean={np.mean(stats[key]):.2f}, std={np.std(stats[key]):.2f}, min={np.min(stats[key]):.2f}, max={np.max(stats[key]):.2f}')

    for key, value in stats.items():
        plt.hist(value, bins=50, density=True)
        plt.title(key)
        plt.xlabel('Value')
        plt.ylabel('Percentage')
        plt.savefig(f"{Path(stat_dir)}/{key}.png")
        plt.clf()

    for key, value in stats.items():
        with open(f"{Path(stat_dir)}/{key}.txt", "w") as f:
            for v in value:
                f.write(f"{v}\n")


if __name__ == '__main__':
    main()


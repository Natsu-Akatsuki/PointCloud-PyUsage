from pathlib import Path
from tqdm import tqdm
import numpy as np

dataset_dir = "/home/helios/mnt/dataset/livox_dataset/training/label_2"
dataset_dir1 = "/home/helios/mnt/dataset/livox_dataset/ImageSets"
dataset_dir = Path(dataset_dir)
output_label_dir = sorted(list(dataset_dir.iterdir()))

valid_label = []

for label_path in tqdm(output_label_dir):
    label_path = dataset_dir / label_path
    with open(label_path, "r") as f:
        lines = f.readlines()
        if len(lines) != 0:
            valid_label.append(label_path.stem)

number = np.array(valid_label, dtype=np.int32)
with open(dataset_dir1 + "/train.txt", "w") as f:
    np.savetxt(f, np.array(valid_label)[number < 6000], fmt="%s")

with open(dataset_dir1 + "/val.txt", "w") as f:
    np.savetxt(f, np.array(valid_label)[number >= 6000], fmt="%s")

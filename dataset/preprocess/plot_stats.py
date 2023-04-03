import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

font_size = 20

output_dir = Path("stats_dir").resolve()
output_dir.mkdir(exist_ok=True)


def plot_stats(class_name):
    def read_data(file_path):
        with open(file_path, 'r') as f:
            return [float(line.strip()) for line in f.readlines()]

    heights1 = read_data(f'kitti_stats/{class_name}_height.txt')
    heights2 = read_data(f'livox_stats/{class_name}_height.txt')
    length1 = read_data(f'kitti_stats/{class_name}_length.txt')
    length2 = read_data(f'livox_stats/{class_name}_length.txt')
    width1 = read_data(f'kitti_stats/{class_name}_width.txt')
    width2 = read_data(f'livox_stats/{class_name}_width.txt')

    # calculate the frequency of each height value
    bins = np.linspace(1.0, 3.0, 22)
    freq1, _ = np.histogram(heights1, bins=bins)
    freq2, _ = np.histogram(heights2, bins=bins)

    # normalize the frequency values to percentages
    freq1 = freq1 / len(heights1) * 100
    freq2 = freq2 / len(heights2) * 100

    # plot the histograms
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), dpi=500)

    axs[0].bar(np.linspace(1.0, 3.0, 21), freq1, width=0.1, align='edge', label='KITTI', color='#8ebad9', alpha=0.8)
    axs[0].bar(np.linspace(1.0, 3.0, 21) + 0.1, freq2, width=0.1, align='edge', label='Livox', color='#ffbe86',
               alpha=0.8)
    axs[0].set_ylabel('Frequency (%)', fontsize=font_size)
    axs[0].set_title('3D Bounding Box Height Distribution', fontsize=font_size)
    axs[0].legend(prop={'size': font_size})

    # calculate the frequency of each length value
    if class_name == 'Car':
        bins = np.linspace(2.0, 8.0, 22)
    elif class_name == 'Pedestrian':
        bins = np.linspace(0, 3.0, 22)
    else:
        bins = np.linspace(1.0, 3.0, 22)
    freq3, _ = np.histogram(length1, bins=bins)
    freq4, _ = np.histogram(length2, bins=bins)

    # normalize the frequency values to percentages
    freq3 = freq3 / len(length1) * 100
    freq4 = freq4 / len(length2) * 100

    # plot the histograms
    axs[1].bar(np.linspace(bins[0], bins[-2], 21), freq3, width=0.1, align='edge', label='KITTI', color='#8ebad9',
               alpha=0.8)
    axs[1].bar(np.linspace(bins[0], bins[-2], 21) + 0.1, freq4, width=0.1, align='edge', label='Livox', color='#ffbe86',
               alpha=0.8)
    axs[1].set_ylabel('Frequency (%)', fontsize=font_size)
    axs[1].set_title('3D Bounding Box Length Distribution', fontsize=font_size)
    axs[1].legend(prop={'size': font_size})

    # calculate the frequency of each width value
    bins = np.linspace(0.5, 2.5, 22)
    freq5, _ = np.histogram(width1, bins=bins)
    freq6, _ = np.histogram(width2, bins=bins)

    # normalize the frequency values to percentages
    freq5 = freq5 / len(width1) * 100
    freq6 = freq6 / len(width2) * 100

    # plot the histograms
    axs[2].bar(np.linspace(0.5, 2.5, 21), freq5, width=0.1, align='edge', label='KITTI', color='#8ebad9', alpha=0.8)
    axs[2].bar(np.linspace(0.5, 2.5, 21) + 0.1, freq6, width=0.1, align='edge', label='Livox', color='#ffbe86',
               alpha=0.8)
    axs[2].set_ylabel('Frequency (%)', fontsize=font_size)
    axs[2].set_title('3D Bounding Box Width Distribution', fontsize=font_size)
    axs[2].legend(prop={'size': font_size})

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=font_size)  # set tick size to 20
    plt.savefig(f'{output_dir}/{class_name}_stats.png')
    plt.show()


for class_name in ['Car', 'Pedestrian', 'Cyclist']:
    plot_stats(class_name)

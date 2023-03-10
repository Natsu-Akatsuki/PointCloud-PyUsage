import argparse
from pathlib import Path

import numpy as np


def load_bin(file_path):
    """
    :param file_path:
    :return:
    """
    assert file_path.split('.')[-1] == 'bin', "The file type is not bin"
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def load_npy(file_path):
    """
    :param file_path:
    :return:
    """
    assert file_path.split('.')[-1] == 'npy', "The file type is not npy"

    pointcloud = np.load(file_path)
    if not isinstance(pointcloud[0][0], np.float32):
        pointcloud = pointcloud.astype(np.float32)
        print("Attention: the pointcloud type will be cast into float32")
    if pointcloud.shape[1] > 4:
        print("Attention: the pointcloud shape is", pointcloud.shape)

    return pointcloud


def load_pcd(file_path):
    """
    :param file_path:
    :return:
    """
    assert file_path.split('.')[-1] == 'pcd', "The file type is not pcd"

    with open(file_path, 'rb') as f:
        for line in range(11):
            lines = f.readline()
            if line == 3:
                # e.g. SIZE 4 4 4
                point_dim = len(lines.decode().strip('\n').split(' ')[1:])
                point_size = np.array(lines.decode().strip('\n').split(' ')[1:], dtype=np.int32).sum()
            if line == 9:
                pts_num = int(lines.decode().strip('\n').split(' ')[-1])
            if line == 10:
                format_type = lines.decode().strip('\n').split(' ')[1]
                if format_type == 'binary':
                    raw_data = f.read(pts_num * point_size)
                    pointcloud = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, point_dim)
                elif format_type == 'ascii':
                    pointcloud = np.loadtxt(f, dtype=np.float32, delimiter=' ')
                else:
                    raise NotImplementedError(f"[IO] The {format_type} pcd file is not supported yet.")

        return pointcloud


def load_txt(file_path):
    """
    :param file_path:
    :return:
    """
    assert file_path.split('.')[-1] == 'txt', "The file type is not txt"
    pointcloud = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
    return pointcloud


def save_pcd(pointcloud_np, file_name="pointcloud.pcd", fields="xyzi"):
    with open(file_name, 'w') as f:
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")

        if fields == "xyzi" or fields == "xyzrgb":
            if fields == "xyzi":
                f.write("FIELDS x y z intensity\n")
            elif fields == "xyzrgb":
                f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        elif fields == "xyz":
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            raise NotImplementedError("The fields is not supported yet.")

        f.write("WIDTH {}\n".format(pointcloud_np.shape[0]))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS {}\n".format(pointcloud_np.shape[0]))

        f.write("DATA binary\n")
        pointcloud_np.tofile(f)

        # ASCII
        # f.write("DATA ascii\n")
        # for i in range(pointcloud.shape[0]):
        #     f.write(
        #         str(pointcloud[i][0]) + " " + str(pointcloud[i][1]) + " " + str(pointcloud[i][2]) + " " + str(
        #             pointcloud[i][3]) + "\n")


def save_pointcloud(pointcloud_np, file_name="pointcloud.npy", fields="xyzi"):
    """
    1) save pointcloud as npy/bin format using numpy API
    2) save pointcloud as pcd format using python API
    :param pointcloud_np:
    :param file_name:
    :param field: only for pcd file
    :return:
    """

    suffix = Path(file_name).suffix[1:]
    if suffix == "npy":
        np.save(file_name, pointcloud_np)
    elif suffix == "bin":
        pointcloud_np.tofile(file_name)
    elif suffix == "pcd":
        save_pcd(pointcloud_np, file_name, fields)
    else:
        raise NotImplementedError


def load_pointcloud(file_name, is_read_only=False):
    assert isinstance(file_name, str), "The file name should be a string"
    suffix = Path(file_name).suffix[1:]
    if suffix == "npy":
        pointcloud = load_npy(file_name)
    elif suffix == "bin":
        pointcloud = load_bin(file_name)
    elif suffix == "pcd":
        pointcloud = load_pcd(file_name)
    else:
        raise NotImplementedError("The file type is not supported yet.")

    if not is_read_only:
        pointcloud = pointcloud.copy()

    return pointcloud


def convert_pointcloud(input_file_name, export_type):
    pc = load_pointcloud(input_file_name)
    output_file_name = str(Path(input_file_name).parent / Path(input_file_name).stem) + ".convert." + f"{export_type}"
    save_pointcloud(pc, output_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", action="store", help="pointcloud file name")
    parser.add_argument('-t', '--type', type=str, default='pcd', help='specify the type of export file')
    args = parser.parse_args()

    convert_pointcloud(args.name, args.type)


if __name__ == '__main__':
    main()

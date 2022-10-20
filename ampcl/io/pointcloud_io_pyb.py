from colorama import Fore, Style

try:
    from pathlib import Path

    from .io_pyb import cLoadPCD


    def c_load_pcd(file_name, remove_nan=True):
        file_name = str(Path(file_name).resolve())
        return cLoadPCD(file_name, remove_nan)


    def c_save_pcd(file_name, pointcloud):
        file_name = str(Path(file_name).resolve())
        io_pyb.save_pcd = io_pyb.savePCD(file_name, pointcloud)

except ImportError as ex:
    print(Fore.YELLOW + f"[IO] {ex.msg}. Using python version only." + Style.RESET_ALL)

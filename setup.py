import os
import shutil
import subprocess
import sys
from distutils.cmd import Command
from distutils.command.clean import clean
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class UninstallCommand(Command):
    description = "uninstall the package and remove the egg-info dir"
    user_options = []

    # This method must be implemented
    def initialize_options(self):
        pass

    # This method must be implemented
    def finalize_options(self):
        pass

    def run(self):
        package_name = "ampcl"
        os.system("pip uninstall -y " + package_name)
        dirs = list((Path('.').glob('*.egg-info')))
        if len(dirs) == 0:
            print('No egg-info files found. Nothing to remove.')
            return

        for egg_dir in dirs:
            shutil.rmtree(str(egg_dir.resolve()))
            print(f"Removing dist directory: {str(egg_dir)}")


class CleanCommand(clean):
    """
    Custom implementation of ``clean`` setuptools command."""

    def run(self):
        """After calling the super class implementation, this function removes
        the dist directory if it exists."""
        self.all = True  # --all by default when cleaning
        super().run()
        if Path('dist').exists():
            shutil.rmtree('dist')
            print("removing 'dist' (and everything under it)")
        else:
            print("'dist' does not exist -- can't clean it")

        if Path('build').exists():
            shutil.rmtree('build')
            print("removing 'build' (and everything under it)")
        else:
            print("'build' does not exist -- can't clean it")


class CMakeExtension(Extension):

    def __init__(self, name, source_dir=""):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.fspath(Path(source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_full_path = Path.cwd() / self.get_ext_fullpath(ext.name)
        ext_dir = ext_full_path.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}/{ext.name.split('.')[1]}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]
        build_args += [f"-j{4}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(['cmake', ext.source_dir] + cmake_args, cwd=str(build_temp), check=True)
        subprocess.run(["cmake", "--build", "."] + build_args, cwd=str(build_temp), check=True)


if __name__ == '__main__':
    setup(
        ext_modules=[CMakeExtension("ampcl.io", source_dir="ampcl/io"),
                     CMakeExtension("ampcl.filter", source_dir="ampcl/filter")],
        cmdclass={'uninstall': UninstallCommand,
                  'clean': CleanCommand,
                  'build_ext': CMakeBuild}
    )

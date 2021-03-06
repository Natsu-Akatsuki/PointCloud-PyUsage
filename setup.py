import os
import shutil
import subprocess
from distutils.cmd import Command
from distutils.command.clean import clean
from pathlib import Path

from setuptools import setup


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
        package_name = "PyPCL"
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


class CMakeBuild(Command):
    description = "perform cmake"
    user_options = []

    # This method must be implemented
    def initialize_options(self):
        self.ext_module = ['PyPCL/pointcloud_io']

    # This method must be implemented
    def finalize_options(self):
        pass

    def run(self):
        for ext in self.ext_module:
            ext = Path(ext).resolve()
            build_path = ext / "build"
            build_path.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(['cmake', '..'], cwd=str(build_path))
            subprocess.check_call(['make', '-j4'], cwd=str(build_path))


if __name__ == '__main__':
    setup(
        cmdclass={'uninstall': UninstallCommand,
                  'clean': CleanCommand,
                  'build_cmake_ext': CMakeBuild}
    )

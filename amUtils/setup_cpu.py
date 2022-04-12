import shutil
from distutils.command.clean import clean
from pathlib import Path
from distutils.cmd import Command
from setuptools import find_packages, setup
import os
import subprocess

package_name = "am-utils"


def get_git_commit_number():
    if not os.path.exists(".git"):
        return "0000000"

    cmd_out = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode("utf-8")[:7]
    return git_commit_number


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
        os.system("pip uninstall -y " + package_name)
        dirs = list((Path('.').glob('*.egg-info')))
        if len(dirs) == 0:
            print('No egg-info files found. Nothing to remove.')
            return

        for egg_dir in dirs:
            shutil.rmtree(str(egg_dir.resolve()))
            print(f"Removing dist directory: {str(egg_dir)}")


def write_version_to_file(version, target_file):
    with open(target_file, "w") as f:
        print('__version__ = "%s"' % version, file=f)


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

with open('requirements.txt') as f:
    required = f.read().splitlines()

if __name__ == '__main__':
    version = "0.0.1+%s" % get_git_commit_number()
    write_version_to_file(version, "version.py")
    setup(
        author='anomynous',
        cmdclass={'uninstall': UninstallCommand,
                  'clean': CleanCommand},
        install_requires=required,
        license="Apache License 2.0",
        name=package_name,
        packages=find_packages(exclude=""),
    )

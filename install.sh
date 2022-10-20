#!/bin/bash
set -e
python3 -m build --wheel -n
pip3 install --user dist/*.whl

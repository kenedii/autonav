#!/bin/bash

# Define the Python site-packages path for system-wide installation (Python 3.6.9 on Jetson)
PYTHON_SITE_PACKAGES_DIR="/usr/local/lib/python3.6/dist-packages"

# 1. Install the core shared library (librealsense2.so) to the system library path
echo "Installing core library to /usr/local/lib/..."
sudo cp librealsense2.so.2.55.1 /usr/local/lib/
sudo ldconfig /usr/local/lib

# 2. Create necessary symlinks for the core library
echo "Creating symlinks..."
cd /usr/local/lib
sudo ln -sf librealsense2.so.2.55.1 librealsense2.so.2.55
sudo ln -sf librealsense2.so.2.55 librealsense2.so
# Note: You need to specify which file the final symlink points to. I've corrected it to link to the shorter name.

# 3. Install the Python binding (pyrealsense2.so) to the Python 3 site-packages
echo "Installing Python bindings to $PYTHON_SITE_PACKAGES_DIR..."
# Note: Ensure you are running this script from the ~/librealsense/build/Release directory or correct the path below.
# This assumes the file is in the same directory as the script run location.
sudo cp ~/librealsense/build/Release/pyrealsense2.cpython-36m-aarch64-linux-gnu.so $PYTHON_SITE_PACKAGES_DIR/pyrealsense2.so

echo "Installation complete. The library should now be permanently accessible."
python3 -c "import pyrealsense2 as rs; print(f'Successfully imported pyrealsense2 version: {rs.__version__}')"


# Setup

This directory contains scripts and artifacts required to prepare a Jetson Nano for this project.

## Platform baseline

- Ubuntu 18.04.6 LTS
- JetPack 4.6.1 SDK
- Python 3.6.9

## Contents

- `install_jupyter.sh`: Installs Jupyter and related dependencies.
- `jetracer_full_build.sh`: Full environment setup script for jetracer and PyTorch dependencies.
- `pyrealsense2/build.sh`: Builds and installs librealsense + pyrealsense2 for Jetson.
- `pyrealsense2/Release/`: Precompiled RealSense shared libraries and python bindings for the baseline platform.
- `pyrealsense2/README.md`: RealSense build details and usage notes.
- `jetracer_scripts_README.md`: Legacy script descriptions that were previously under `jetracer/`.

## Suggested setup flow

1. Run `install_jupyter.sh` for notebook workflow support.
2. Run `jetracer_full_build.sh` for core car dependencies.
3. Build RealSense with `pyrealsense2/build.sh` if compiling locally.
4. Optionally use `pyrealsense2/Release/librealsense2_pythonsetup.sh` to install prebuilt python artifacts for the baseline environment.

## Notes

- The release binaries target ARM Cortex-A57 Jetson Nano devices with the exact baseline listed above.
- If your JetPack, Ubuntu, or Python versions differ, rebuild from source instead of using precompiled binaries.

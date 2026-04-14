#!/bin/bash

# --- Configuration and Setup ---
PIP_EXE=pip3 
PYTHON_EXE=/usr/bin/python3 

INSTALL_DIR=~/jetson_ai_tools
TORCH_VERSION="1.10.0" 
TV_VERSION="v0.11.1" 
PILLOW_VERSION="7.0.0" 

echo "Starting FINAL Resilient Jetson AI-IOT Tools Installation (v5 - Pillow Dependencies Fixed)..."
echo "Target PyTorch Version: ${TORCH_VERSION}, Pillow Version: ${PILLOW_VERSION}"

# Exit immediately if a command exits with a non-zero status
set -e

# --- Utility Functions ---

# Function to check if a python module is installed
is_installed() {
    "${PYTHON_EXE}" -c "import $1" 2>/dev/null
}

# Function for robust PyTorch download
download_pytorch() {
    local URL_ARRAY=("https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl" \
                     "https://nvidia.box.com/shared/static/ryfclgvkkuo03m0x7d6h4llnmc54zog8.whl")
    local WHL_FILE="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    
    echo "Checking for PyTorch wheel file..."
    if [ -f "${WHL_FILE}" ] && [ $(stat -c%s "${WHL_FILE}") -ge 100000000 ]; then
        echo "PyTorch wheel already exists and looks complete."
        return 0
    fi

    echo "Removing any incomplete wheel and attempting download..."
    rm -f "${WHL_FILE}"
    
    for URL in "${URL_ARRAY[@]}"; do
        echo "Attempting download from: ${URL}"
        if curl -L --retry 5 "${URL}" -o "${WHL_FILE}"; then
            if [ $(stat -c%s "${WHL_FILE}") -ge 100000000 ]; then
                echo "Download successful."
                return 0
            else
                echo "Download appears incomplete. Trying next URL."
                rm -f "${WHL_FILE}"
            fi
        fi
    done

    echo "ERROR: Failed to download PyTorch wheel after all attempts."
    exit 1
}

# --- 1. Cleanup and System Setup ---
echo "--- 1. Cleanup, System Dependencies (Including Pillow Fix), and Path Conflict Resolution ---"

# --- CRITICAL FIX: Clean up local Python install directory ---
USER_SITE_PACKAGES="/home/jetson/.local/lib/python3.6"
if [ -d "${USER_SITE_PACKAGES}" ]; then
    echo "🚨 Deleting conflicting user site-packages directory: ${USER_SITE_PACKAGES}"
    sudo rm -rf "${USER_SITE_PACKAGES}"
fi

# Install all required system libraries
sudo apt update
# --- PILLOW FIX: Added libjpeg-dev, zlib1g-dev, libtiff-dev, libfreetype6-dev ---
sudo apt install -y build-essential git cmake curl \
    python3-dev python3-pip libopenblas-dev libffi-dev \
    libjpeg-dev zlib1g-dev libtiff-dev libfreetype6-dev

echo "Upgrading pip3 and installing core Python build packages globally..."
sudo -H "${PIP_EXE}" install --upgrade pip
sudo -H "${PIP_EXE}" install --upgrade setuptools wheel packaging Cython

# --- 2. PyTorch and TorchVision ---
echo "--- 2. Checking and Installing PyTorch & TorchVision ---"

if ! is_installed torch; then
    echo "PyTorch not found. Beginning installation..."
    download_pytorch
    
    echo "Installing PyTorch ${TORCH_VERSION}..."
    sudo -H "${PIP_EXE}" install ./torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    rm -f ./torch-1.10.0-cp36-cp36m-linux_aarch64.whl
else
    echo "PyTorch is already installed. Skipping download/install."
fi

if ! is_installed torchvision; then
    echo "TorchVision not found. Beginning installation..."
    
    mkdir -p "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
    
    # 2a. Install compatible Pillow version (now the necessary system headers are installed)
    echo "Installing compatible Pillow version (${PILLOW_VERSION})..."
    sudo -H "${PIP_EXE}" install "Pillow==${PILLOW_VERSION}"

    # 2b. Clone and build TorchVision
    if [ -d "vision" ]; then rm -rf vision; fi
    
    TV_REPO_PRIMARY="https://github.com/pytorch/vision"
    echo "Cloning TorchVision ${TV_VERSION}..."
    git clone --branch "${TV_VERSION}" "${TV_REPO_PRIMARY}" vision

    cd vision
    export BUILD_VERSION=${TV_VERSION}
    echo "Building and installing TorchVision..."
    # Note: Pillow is now installed, allowing the torchvision build to use PIL
    sudo "${PYTHON_EXE}" setup.py install
    cd ..
else
    echo "TorchVision is already installed."
fi

# --- 3. NVIDIA AI-IOT Libraries ---
echo "--- 3. Checking and Installing NVIDIA AI-IOT Libraries (JetCam, torch2trt, JetRacer) ---"
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# 3a. Install JetCam
if ! is_installed jetcam; then
    echo "Installing JetCam..."
    if [ -d "jetcam" ]; then rm -rf jetcam; fi
    git clone https://github.com/NVIDIA-AI-IOT/jetcam
    cd jetcam
    sudo "${PYTHON_EXE}" setup.py install
    cd ..
else
    echo "JetCam is already installed."
fi

# 3b. Install torch2trt
if ! is_installed torch2trt; then
    echo "Installing torch2trt..."
    if [ -d "torch2trt" ]; then rm -rf torch2trt; fi
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo "${PYTHON_EXE}" setup.py install
    cd ..
else
    echo "torch2trt is already installed."
fi

# 3c. Install JetRacer
if ! is_installed jetracer; then
    echo "Installing JetRacer..."
    if [ -d "jetracer" ]; then rm -rf jetracer; fi
    git clone https://github.com/NVIDIA-AI-IOT/jetracer
    cd jetracer
    
    echo "Installing remaining JetRacer Python dependencies (traitlets, ipywidgets)..."
    sudo -H "${PIP_EXE}" install traitlets ipywidgets
    
    echo "Building and installing JetRacer..."
    sudo "${PYTHON_EXE}" setup.py install

    echo "Enabling Jupyter widgets extension (needed for notebooks)..."
    sudo jupyter nbextension enable --py widgetsnbextension
    cd ..
else
    echo "JetRacer is already installed."
fi


# --- 4. Final Verification and Cleanup ---
echo "--- 4. Final Verification and Cleanup ---"

# Clean up the installation directory (cloned repos)
echo "Cleaning up temporary build files in ${INSTALL_DIR}..."
sudo rm -rf "${INSTALL_DIR}" 

echo "Final Verification of all imports:"
# Verification checks
${PYTHON_EXE} -c "
import torch
import torchvision
import torch2trt
import jetcam
import jetracer
import PIL
print('✅ Success: All core packages imported without error.')
import PIL.Image
print(f'PyTorch Version: {torch.__version__}')
print(f'TorchVision Version: {torchvision.__version__}')
print(f'Pillow Version: {PIL.__version__}')
"

echo " "
echo "✨ Installation is complete! Please **reboot your Jetson Nano** now for configuration stability before using the notebooks."


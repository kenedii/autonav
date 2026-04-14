#!/bin/bash
# FULL RealSense + pyrealsense2 installer for Jetson Nano (2025)
# Includes symlink fixes + Python binding installation
# Works on JetPack 4.6 (Python 3.6.9)

set -e  # stop on error

echo "===================================================="
echo " Jetson Nano — RealSense FULL INSTALLER (2025 READY)"
echo "===================================================="

### 1. SYSTEM UPDATE + DEPENDENCIES
echo "[1/8] Installing dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev python3-pip python3-setuptools qt5-default qtbase5-dev

### 2. CREATE SWAP
echo "[2/8] Creating 6GB swapfile..."
if ! grep -q "/swapfile" /etc/fstab ; then
    sudo fallocate -l 6G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
else
    echo "Swap already exists — skipping."
fi

### 3. CLONE LIBREALSENSE
echo "[3/8] Cloning RealSense repo..."
cd $HOME
rm -rf librealsense
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1  # stable version tested on Nano

### 4. UDEV RULES
echo "[4/8] Applying udev rules..."
sudo ./scripts/setup_udev_rules.sh

### 5. BUILD LIBREALSENSE (SO + VIEWER + PYTHON)
echo "[5/8] Building libraries (takes ~60 mins)..."
mkdir -p build && cd build
cmake ../ \
    -DBUILD_PYTHON_BINDINGS=bool:true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=true

make -j2
sudo make install
sudo ldconfig

### 6. INSTALL SHARED OBJECTS (.so)
echo "[6/8] Installing shared libraries..."

RELEASE_DIR="$HOME/librealsense/build/Release"
CORE_SO=$(find "$RELEASE_DIR" -maxdepth 1 -name "librealsense2.so*" | head -n 1)

if [ -f "$CORE_SO" ]; then
    echo "Copying core RealSense .so → /usr/local/lib/"
    sudo cp "$CORE_SO" /usr/local/lib/
    cd /usr/local/lib
    sudo ln -sf "$(basename $CORE_SO)" librealsense2.so
    sudo ln -sf "$(basename $CORE_SO)" librealsense2.so.2.55
else
    echo "WARNING: librealsense2.so NOT FOUND!"
fi

### 7. INSTALL PYTHON BINDINGS
echo "[7/8] Installing Python bindings..."
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_SITE="/usr/local/lib/python${PYVER}/dist-packages"

sudo mkdir -p "$PY_SITE"
PY_SO=$(find "$RELEASE_DIR" -maxdepth 1 -name "pyrealsense2*.so" | head -n 1)

if [ -f "$PY_SO" ]; then
    echo "Copying pyrealsense2.so → $PY_SITE"
    sudo cp "$PY_SO" "$PY_SITE/pyrealsense2.so"
    echo "export PYTHONPATH=\$PYTHONPATH:$PY_SITE" >> ~/.bashrc
else
    echo "ERROR: pyrealsense2.so NOT FOUND!"
    exit 1
fi

sudo ldconfig

### 8. TEST & VERIFY INSTALL
echo "[8/8] Testing pyrealsense2 import..."
python3 - << EOF
try:
    import pyrealsense2 as rs
    print("[OK] pyrealsense2 imported — Version:", rs.__version__)
except Exception as e:
    print("[FAIL] Could not import pyrealsense2:", e)
EOF

echo ""
echo "===================================================="
echo " 🎉 INSTALLATION COMPLETE — REBOOT NOW:"
echo "     sudo reboot"
echo ""
echo "TEST AFTER REBOOT:"
echo "    rs-enumerate-devices    # list cameras"
echo "    realsense-viewer        # GUI viewer"
echo "    python3 -c \"import pyrealsense2 as rs; print(rs.context().devices)\""
echo "===================================================="

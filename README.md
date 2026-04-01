# Jetson Nano Racer – Autonomous RC Car with RealSense D435i

This project deploys a deep learning lane-following model on an
**NVIDIA Jetson Nano** mounted on a **LaTrax 1/18 RC car**. CAM0 is the
primary forward RGB source for lane following and AprilTags, while the
**Intel RealSense D435i** remains active as a sidecar for depth-stop and
IMU context. CAM1 is reserved for rear-preview / reverse-only scaffolding.
Steering commands are sent to the PCA9685 servo driver in real time.

## Hardware

- Jetson Nano 4GB Developer Kit with ARM Cortex-A57 CPU, Ubuntu + JetPack
- LaTrax Rally 1/18 RC car
- Intel RealSense D435i (sidecar depth + IMU)
- Front CAM0 fisheye camera for primary forward RGB
- Rear CAM1 camera for preview / reverse-only support
- TP-Link TL-WN725N USB WiFi adapter
- PCA9685 16-channel servo driver
- Pololu 4-Channel RC Servo Multiplexer
- Fan-4020-PWM-5V
- XBOX Controller
- Batteries, mounts, cabling

## Software

- Ubuntu 18.04.6 LTS
- Jetpack 4.6.1 SDK
- Python 3.6.9

## Model Architecture

- Supports several Resnet variants: ```Resnet18, Resnet34, Resnet50, Resnet101```

## Core Features

- Data collection from teleoperated driving
- ResNet-based steering model (PyTorch + TensorRT)
- Autonomous lane following on indoor track
- CAM0 fisheye preview and lane-follow training pipeline
- RealSense depth ROI measurement for safety / debugging
- REST API for live predictions
- Host dashboard for fleet/operator monitoring
- Dockerfile support for deployment workflows

## Expo Demo Mode / AutoNav Slice

This repository also supports an expo-ready AutoNav slice layered on top of the existing lane follower.

- Default route name: `expo_route`
- Default checkpoint tags: `start/home = 10`, `checkpoint = 20`, `goal = 30`
- Ordered behavior: the route starts in `RUNNING`, tag `20` marks checkpoint progress, and tag `30` only counts after the checkpoint has been seen
- Depth-based obstacle stop is independent of YOLO and uses the RealSense front ROI
- Obstacle stops require a manual operator restart for safety
- SLAM is not required for demo mode
- If AprilTag support is unavailable at runtime, the car keeps lane following and reports tag detection as unavailable
- CAM0 is the forward preview and primary training source in the recommended configuration
- CAM1 does not participate in ordinary forward control
- Legacy single-camera configs still work; role-based camera configs are preferred for new runs
- The CAM0 training/inference profile is `cam0_fisheye_v1`; older RealSense RGB runs remain on the legacy resize path

Recommended camera config shape:

```json
[
  {
    "role": "primary_rgb",
    "type": "csi",
    "sensor_id": 0,
    "width": 640,
    "height": 480,
    "fps": 15,
    "flip_method": 2,
    "enabled": true
  },
  {
    "role": "sidecar_depth_imu",
    "type": "realsense",
    "width": 640,
    "height": 480,
    "fps": 15,
    "enabled": true
  },
  {
    "role": "rear_preview",
    "type": "csi",
    "sensor_id": 1,
    "width": 640,
    "height": 480,
    "fps": 15,
    "flip_method": 2,
    "enabled": false
  }
]
```

## Development Workflow

Use Git as the source of truth for source changes and treat the Jetson as a
deployment target that pulls tested code from Git. The workflow reference lives
in [docs/git_workflow.md](docs/git_workflow.md).

## 1. Setup

### 1.1 Jetson Base Setup

1. Flash JetPack and boot Nano.
2. Install system packages:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip git

If the car is not moving when model is running, run ```sudo bash -c 'i2cset -y 1 0x40 0x00 0x21; i2cset -y 1 0x40 0xFE 0x65; i2cset -y 1 0x40 0x00 0xA1; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x09 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 1; i2cset -y 1 0x40 0x0C 0x00 0x09 && sleep 4; i2cset -y 1 0x40 0x0C 0x00 0x06; echo "FINISHED"'``` (directly writes raw register values via I2C to wake up the PCA9685, set it to 50 Hz, sweep the steering servo fully left → right → center, slam the throttle channel to full forward for 4 seconds, then return everything to neutral)

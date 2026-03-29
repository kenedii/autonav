# Jetson Nano Racer – Autonomous RC Car with RealSense D435i

This project deploys a deep learning lane-following model on an
**NVIDIA Jetson Nano** mounted on a **LaTrax 1/18 RC car** using an
**Intel RealSense D435i** depth camera. The model predicts steering
commands from the RGB stream in real time; control is sent to the
PCA9685 servo driver to steer the car.

## Hardware

- Jetson Nano 4GB Developer Kit with ARM Cortex-A57 CPU, Ubuntu + JetPack
- LaTrax Rally 1/18 RC car
- Intel RealSense D435i (RGBD + IMU)
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
- RealSense center-depth measurement for safety / debugging
- REST API for live predictions
- Streamlit demo UI
- Dockerfile for Streamlit deployment

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

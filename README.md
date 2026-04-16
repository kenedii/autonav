# AutoNav – AI Autonomous RC Car End-to-End Pipeline for NVIDIA Jetson Nano or Rockchip NPU Devices

This project presents an end-to-end pipeline for data collection, model training, and model inference for a deep learning self-driving lane-following model.

It allows you to take a Single Board Computer with CUDA cores or a Rockchip NPU, a camera, and a RC Car, and:
- Collect Data by driving the car around a track made of tape or another material.
- Use a Data Management Frontend to clean your data and generate augmented images.
- Train a model to predict steering and/or throttle values
  - Resnet CNN Backbone to extract features from RGB images
  - MLP Regressor to predict normalized steering and/or throttle values from extracted features
- Run the model and watch the car autonomously drive around your track
- Use the Fleet Management Frontend to remotely start/stop vehicle, load new models, camera live view, and view logs.

The pipeline has been optimized for a **NVIDIA Jetson Nano** mounted on a **LaTrax 1/18 RC car** using CUDA inference, or a **Raxda Rock 5B** with a generic toy RC car using RKNN inference. 

## Top-level layout

- `tests/`: Hardware checks and test scripts.
- `setup/`: Jetson setup and build scripts, including bundled RealSense artifacts.
- `inference/`: Model optimization and on-car autonomous runtime scripts.
- `data_collection/`: Data recording, dataset management frontend, and augmentation utilities.
- `model_training/`: Model training code for both legacy RGB-only and new sensor-combination workflows.
- `fleet/`: Fleet-facing client + host application code.

## Quick start path

1. Start with setup docs in `setup/README.md`.
2. Verify controls and hardware using scripts in `tests/README.md`.
3. Collect data and interact with Data Management Frontend using `data_collection/README.md`.
4. Train models from `model_training/README.md`.
5. Optimize/deploy with `inference/README.md`.
6. Run fleet workflows, manage cars from Fleet Management Frontend with `fleet/fleet_management_app/README.md`.


## Hardware

**Jetson Nano Prototype (Sensor-Fusion with Depth, IR, and 360 degree FOV)**
- Jetson Nano 4GB Developer Kit with ARM Cortex-A57 CPU and Fan-4020-PWM-5V, Ubuntu + JetPack
- LaTrax Rally 1/18 RC car
- Intel RealSense D435i (sidecar depth + IMU)
- Front CAM0 fisheye camera for primary forward RGB
- Rear CAM1 camera for preview / reverse-only support
- TP-Link TL-WN725N USB WiFi adapter
- PCA9685 16-channel servo driver
- Pololu 4-Channel RC Servo Multiplexer
- Batteries, mounts, cabling

After model inference, the Jetson Nano outputs steering-angle and throttle predictions. It sends these values over I²C to the PCA9685 servo driver (configured at 50 Hz), which converts them into standard RC PWM pulses (pulse-width in microseconds). The PCA9685 then feeds the PWM signals through the Pololu 4-channel RC servo multiplexer directly to the LaTrax car’s steering servo and electronic speed controller (ESC)

**Radxa Rockchip 5B Prototype (Budget Model, Cheapest Proof-Of-Concept)**
- Radxa Rock 5B with Rockchip RK3588 SoC and Radxa Heatsink 4012, Rock 5B Armbian
- Raspberry Pi Pico
- Generic $10 Toy RC Car (WalMart)
- L298N motor driver module
- TP-Link TL-WN725N USB WiFi adapter
- Generic $5 USB Webcam
- Batteries, mounts, cabling

The Rock 5B runs the same inference model and sends the resulting steering/throttle commands (desired PWM pulse widths or motor speeds) over serial/USB to the Raspberry Pi Pico. The Pico then generates precise PWM signals in hardware and drives the L298N motor-driver module, which controls direction and speed of the two DC motors in the toy RC car.

Needed for data collection:
- XBOX Controller
To collect data by manually driving the car around the track, you must have a game controller. We configured it to use a $5 USB XBOX Controller, however other controllers may work.

CAM0 is the primary forward RGB source for lane following, while the **Intel RealSense D435i** remains active as a sidecar for depth-stop and IMU context. CAM1 is reserved for rear-preview / reverse-only scaffolding. Steering commands are sent to the PCA9685 servo driver in real time.

## Software

- Ubuntu 18.04.6 LTS
- Jetpack 4.6.1 SDK
- Python 3.6.9

## Core Features

- Data collection from teleoperated driving
- ResNet-based steering model (PyTorch + TensorRT)
- Autonomous lane following on indoor track
- CAM0 fisheye preview and lane-follow training pipeline
- RealSense depth ROI measurement for safety / debugging
- REST API for live predictions
- Host dashboard for fleet/operator monitoring
- Dockerfile support for deployment workflows

## Model Architecture

- Supports several Resnet variants: ```Resnet18, Resnet34, Resnet50, Resnet101, Resnet152```

The project defines multiple model variants through a list called EXPERIMENTS. This allows easy training and evaluation of different sensor combinations without changing the core training code.
``` EXPERIMENTS = [
    {"id": 1, "desc": "Front+Back + all sensors", "csv": AUGMENTED_CSV, "features": ['rgb_path', 'cam1_path', 'ir_path', 'depth_path']},
    {"id": 2, "desc": "Front only + all sensors", "csv": AUGMENTED_CSV, "features": ['rgb_path', 'ir_path', 'depth_path']},
    {"id": 3, "desc": "Front only RGB only",     "csv": AUGMENTED_CSV, "features": ['rgb_path']},
    {"id": 4, "desc": "Front+Back RGB only",      "csv": AUGMENTED_CSV, "features": ['rgb_path', 'cam1_path']},
    {"id": 5, "desc": "Front+Back + all sensors (Cleaned)",   "csv": CLEANED_CSV,   "features": ['rgb_path', 'cam1_path', 'ir_path', 'depth_path']},
    {"id": 6, "desc": "Front+Back RGB only (Cleaned)",        "csv": CLEANED_CSV,   "features": ['rgb_path', 'cam1_path']}
]
```
Experiment 5 and 6 are identical to 1 and 2 respectively, these were just created these to do a training run with non-augmented images only, so they can be ignored. (Using no augmented images performs much worse)

- rgb_path: Front camera on vehicle
- cam1_path: Back camera on vehicle
- IR_path: File path of IR image from Realsense Camera
- Depth_path: File path of Depth map image from Realsense Camera

## Pre-trained Model Weights

Several pre-trained models are available [from our Huggingface repository](https://huggingface.co/everestt/autonav/tree/main).

The pre-trained AutoNav Models we have available are:
- AutoNav v1 (Best model: AutoNav-v1-34: Steering Pseudo Accuracy of 72.70%)
  - Predicts normalized steering value from RGB image
- AutoNav v2 (Best model: AutoNav-v2-34: Steering Pseudo Accuracy of 94.20%)
  - Predicts normalized steering and throttle values from RGB image

The Steering Pseudo Accuracy evaluation metric sorts validation images into [Left, Centre, Right] bins and evaluates accuracy to predict a normalized steering value within the correct bin.

Some model training runs were done with a capped throttle value for safety reasons, so it may not predict high throttle values. To convert the normalized throttle prediction from the model output to a [-1.0, 1.0] range, apply the formula: 
- **new_norm_throttle = max(-1.0, min(1.0, model_output × 3.33))**

*Note: Our pretrained steering models predict +1.0 for left and -1.0 for right. The output value may need to be inverted (multiply by -1) depending on the car motor driver module.*

## Demos 

### Training Data Example (Post-Augmentations)
<img width="1650" height="560" alt="augmented_data_train_samples_by_source_examples" src="https://github.com/user-attachments/assets/5bdd33dd-3efb-4e96-9a43-299e4e838777" />
The images from the top dataset are used in AutoNav v1 Models only.

### AutoNav V1 Live Demo

https://github.com/user-attachments/assets/af635698-d848-48ed-9db1-3eb8aa4ac871

## Troubleshooting

If the car is not moving when model is running, run ```sudo bash -c 'i2cset -y 1 0x40 0x00 0x21; i2cset -y 1 0x40 0xFE 0x65; i2cset -y 1 0x40 0x00 0xA1; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x09 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 1; i2cset -y 1 0x40 0x0C 0x00 0x09 && sleep 4; i2cset -y 1 0x40 0x0C 0x00 0x06; echo "FINISHED"'``` (directly writes raw register values via I2C to wake up the PCA9685, set it to 50 Hz, sweep the steering servo fully left → right → center, slam the throttle channel to full forward for 4 seconds, then return everything to neutral)

This worked to "warm up" the PCA9685 so the model inference code could run properly.

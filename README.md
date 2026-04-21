# AutoNav – AI Autonomous RC Car End-to-End Pipeline for NVIDIA Jetson Nano or Rockchip NPU Devices

This project presents an end-to-end pipeline for data collection, model training, and model inference for a deep learning self-driving lane-following model.

It allows you to take a Single Board Computer with CUDA cores or a Rockchip NPU, a camera, and a RC Car, and:
- Collect Data by driving the car around a track made of tape or another material.
- Use a Data Management Frontend to clean your data and generate augmented images.
- Train a model to predict steering and/or throttle values
  - Resnet CNN Backbone to extract features from RGB images
  - MLP Regressor with Tanh head to predict normalized steering and/or throttle values from extracted features
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

## Final Presentation / Known-Good State

This repository's known-good final presentation state is:

- Git commit: `9db24f0778a5fc6e02ca1f3eb6a4681f8ced0b95`
- Branch used for the final technical review: `main`
- Validated live demo path: Jetson Nano + CAM0 + TensorRT + PCA9685 + `inference/run_autonomous_resnet.py`

For a concise final-presentation summary, see [docs/final_presentation_status.md](docs/final_presentation_status.md).

## Reproduce from GitHub

Use this checklist to reproduce the project at a high level:

1. Clone the repository.
2. Set up the Jetson environment from `setup/README.md`.
3. Download the model weights and place them in the expected `checkpoints/` path.
4. Verify camera, controller, and motor-control hardware using `tests/README.md`.
5. Collect data with the recorder scripts in `data_collection/`.
6. Train/evaluate a model with `model_training/`.
7. Optimize the selected checkpoint with TensorRT from `inference/`.
8. Run the validated live Jetson demo command below.
9. Launch the fleet dashboard/client if needed.
10. Run the targeted regression/API tests listed in `Run the Tests`.

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

When recording with `--camera realsense --record_mode all`, `depth_path` stores aligned raw `uint16` depth PNGs in millimeters for replay/training. New runs also persist aligned full-resolution RealSense RGB (`realsense_rgb_path`) plus accel/gyro vectors and stream timestamps in `dataset.csv`. Older runs may still contain colorized preview PNGs and no IMU columns.

## Software

- Ubuntu 18.04.6 LTS
- Jetpack 4.6.1 SDK
- Python 3.6.9

## Known-Good Platform

The final presentation validation path was exercised on:

- Ubuntu `18.04.6`
- JetPack `4.6.1`
- Python `3.6.9`
- Jetson Nano `4GB`
- CAM0 primary RGB input
- PCA9685 motor control over I2C
- Intel RealSense D435i as a sidecar / experimental RGB-D/IMU path

## Feature Status

- Lane following: validated live demo path
  - Primary path is CAM0 + AutoNav-v2-34 + TensorRT + PCA9685 + `inference/run_autonomous_resnet.py`
- YOLO: prototype / advisory only
  - Detection path exists in the fleet runtime
  - Not part of the validated live Jetson control loop
- SLAM: experimental RGB-D odometry / replay only
  - Replay and pose-estimation evidence exists
  - This is not full production SLAM and is not the validated live demo path
- Depth stop: subsystem / prototype only
  - RealSense depth utilities exist
  - This should not be presented as a validated final live-demo safety feature unless re-tested separately

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

The validated final Jetson demo path expects the AutoNav v2 checkpoint at:

`checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth`

Direct Hugging Face path for that checkpoint family:

`https://huggingface.co/everestt/autonav/tree/main/AutoNav-v2/AutoNav-v2-34`

Model weights are not committed to Git in this repository. Download them separately and place them at the expected local destination above.

The pre-trained AutoNav Models we have available are:
- AutoNav v1 (Best model: AutoNav-v1-34: Steering Pseudo Accuracy of 72.70%)
  - Predicts normalized steering value from RGB image
- AutoNav v2 (Best model: AutoNav-v2-34: Steering Pseudo Accuracy of 94.20%)
  - Predicts normalized steering and throttle values from RGB image

The Steering Pseudo Accuracy evaluation metric sorts validation images into [Left, Centre, Right] bins and evaluates accuracy to predict a normalized steering value within the correct bin.

The pseudo-accuracy values above are historical reported metrics from project training runs. If you are presenting or reproducing the repo from GitHub, treat them as reported results unless you also regenerate the supporting training artifacts locally.

Some model training runs were done with a capped throttle value for safety reasons, so it may not predict high throttle values. To convert the normalized throttle prediction from the model output to a [-1.0, 1.0] range, apply the formula: 
- **new_norm_throttle = max(-1.0, min(1.0, model_output × 3.33))**

*Note: Our pretrained steering models predict +1.0 for left and -1.0 for right. The output value may need to be inverted (multiply by -1) depending on the car motor driver module.*

## Validated Jetson Live Demo Command

This is the primary validated final-presentation live-demo path:

```bash
python3 inference/run_autonomous_resnet.py \
  --arch resnet34 \
  --exp 3 \
  --camera cam0 \
  --cam-backend argus \
  --cam-sensor-id 0 \
  --controller-backend pca9685 \
  --model-path checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth \
  --trt-model-path inference/best_model_trt.pth \
  --throttle 0.20 \
  --no-invert-steering \
  --debug-timings
```

This command reflects the final presentation state:

- CAM0 is the primary forward RGB source
- AutoNav-v2-34 / ResNet34 is the active control model
- TensorRT is the validated Jetson optimization path
- PCA9685 is the validated motor-control path
- RealSense is not required for the core live lane-following demo

## Run the Tests

The following targeted test commands were used in the final technical review:

```bash
# AutoNav v2 model/runtime loading checks
pytest fleet/fleet_management_app/client_api/test_autonav_v2.py -q

# SLAM core and replay checks
pytest tests/test_slam_core.py tests/test_slam_replay.py -q

# Fleet API tests
PYTHONPATH=fleet/fleet_management_app/client_api:fleet/fleet_management_app/host_app pytest tests/test_server.py tests/test_client.py -q

# Mission-state tests
PYTHONPATH=fleet/fleet_management_app/client_api pytest tests/test_mission.py -q

# Preprocess/profile tests
PYTHONPATH=data_collection pytest tests/test_preprocess_utils.py -q
```

Known limitation:

- `tests/test_runtime_split.py` reflects an older runtime split and was stale against the final reviewed repo state. Do not present it as part of the passing final validation set unless you update it separately.

## Team Contributions

This project has two members:

- Nick
- Chris

For the final presentation, contributions should be described by subsystem familiarity rather than exclusive authorship. The Git history is mixed across shared files, so the safest summary is:

- Nick
  - hardware / Jetson integration
  - Jetson-side runtime bring-up and live-demo validation
  - familiarity with the Jetson-side prototype/runtime integrations used during final testing

- Chris
  - model training workflow
  - dashboard / API and broader shared codebase implementation
  - major contributor to the repository foundation

- Shared / collaborative areas
  - data collection
  - deployment support
  - testing / reproducibility / demo support

During code review, present ownership by module familiarity rather than claiming strict one-person ownership of every file.

## Demos 

### Training Data Example (Post-Augmentations)
<img width="1650" height="560" alt="augmented_data_train_samples_by_source_examples" src="https://github.com/user-attachments/assets/5bdd33dd-3efb-4e96-9a43-299e4e838777" />
The images from the top dataset are used in AutoNav v1 Models only.

### AutoNav V1 Live Demo

https://github.com/user-attachments/assets/af635698-d848-48ed-9db1-3eb8aa4ac871

## Troubleshooting

If the car is not moving when model is running, run ```sudo bash -c 'i2cset -y 1 0x40 0x00 0x21; i2cset -y 1 0x40 0xFE 0x65; i2cset -y 1 0x40 0x00 0xA1; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x09 && sleep 2; i2cset -y 1 0x40 0x08 0x00 0x06 && sleep 1; i2cset -y 1 0x40 0x0C 0x00 0x09 && sleep 4; i2cset -y 1 0x40 0x0C 0x00 0x06; echo "FINISHED"'``` (directly writes raw register values via I2C to wake up the PCA9685, set it to 50 Hz, sweep the steering servo fully left → right → center, slam the throttle channel to full forward for 4 seconds, then return everything to neutral)

If the car is not moving when model is running, run ```sudo bash -c 'i2cset -y 1 0x40 0x00 0x21; i2cset -y 1 0x40 0xFE 0x65; i2cset -y 1 0x40 0x00 0xA1; i2cset -y 1 0x40 0x08 0x0600 w && sleep 2; i2cset -y 1 0x40 0x08 0x0900 w && sleep 2; i2cset -y 1 0x40 0x08 0x0600 w && sleep 1; i2cset -y 1 0x40 0x0C 0x0900 w && sleep 4; i2cset -y 1 0x40 0x0C 0x0600 w; echo "FINISHED"'``` (Jetson-compatible word writes that wake up the PCA9685, set it to 50 Hz, sweep the steering servo fully left → right → center, slam the throttle channel to full forward for 4 seconds, then return everything to neutral)

This worked to "warm up" the PCA9685 so the model inference code could run properly.

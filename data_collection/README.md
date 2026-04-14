# Data Collection

This folder contains the full pipeline for recording driving data, preparing datasets, and managing collected runs before training.

## What this folder is for

- Record synchronized steering and throttle labels while driving manually.
- Capture camera sensor streams from RealSense and/or OpenCV cameras.
- Build and clean dataset CSVs for downstream training scripts.
- Manage datasets and run augmentation through the web frontend tools.

## Main scripts

- `record_data.py`: Baseline recorder that captures RGB + IR + depth with controller labels.
- `record_data2.py`: Lower-latency recorder that keeps a reduced payload to avoid controller lag.
- `record_data3.py`: Network-controller recorder using UDP controller input from another machine.
- `net_controller_client.py`: Companion client script that reads controller values with pygame and streams axes to the Jetson.
- `realsense_full.py`: Shared RealSense helpers used by recorder scripts.
- `realsense_cv2.py`: OpenCV-style RealSense camera utilities.
- `realsense_pyrealsense2.py`: pyrealsense2 camera interface utilities.

## Requirements files

- `requirements.txt`: General dependencies for recording pipeline scripts.
- `camera_requirements.txt`: Camera-related dependencies used by camera scripts.
- `realsense_requirements.txt`: Dependencies specific to RealSense integration.

## Data frontend

The `data_frontend/` folder provides a dataset management and augmentation web application.

- `data_frontend/app.py`: Main backend/API for managing runs and dataset operations.
- `data_frontend/augment_data.py`: CPU augmentation pipeline.
- `data_frontend/augment_data_gpu.py`: GPU-accelerated augmentation path.
- `data_frontend/dataset_csv_creator.py`: CSV generation utility for training metadata.
- `data_frontend/model_api/resnet_api.py`: Model-serving endpoints used by frontend tools.
- `data_frontend/tools/data_management.py`: Dataset indexing and lifecycle helpers.
- `data_frontend/tools/model_prediction.py`: Utilities for model inference during dataset review.
- `data_frontend/gcp_deploy/`: Container and deployment files for hosted frontend/API.

See `data_frontend/README.md` and `data_frontend/tools/README.md` for deeper implementation details.

## Camera mode support

Recorder scripts support both RealSense and OpenCV paths.

- Use `--camera realsense` for Intel RealSense capture.
- Use `--camera opencv --device <id>` for USB camera capture.

Example:

```bash
python data_collection/record_data3.py --camera realsense
python data_collection/record_data3.py --camera opencv --device 0
```

## Legacy reference

Historical notes from the original layout are preserved in `training_collection_README.md`.

# Jetson Nano Racer

This repository has been reorganized by workflow so it is easier to onboard new users and maintain the RC stack over time.

## Top-level layout

- `tests/`: Hardware checks and test scripts.
- `setup/`: Jetson setup and build scripts, including bundled RealSense artifacts.
- `inference/`: Model optimization and on-car autonomous runtime scripts.
- `data_collection/`: Data recording, dataset management frontend, and augmentation utilities.
- `model_training/`: Model training code for both legacy RGB-only and new sensor-combination workflows.
- `fleet/`: Fleet-facing client + host application code.

## Hardware and software baseline

- Jetson Nano 4GB Developer Kit
- Ubuntu 18.04.6 LTS
- JetPack 4.6.1 SDK
- Python 3.6.9
- Intel RealSense D435i sidecar camera

## Quick start path

1. Start with setup docs in `setup/README.md`.
2. Verify controls and hardware using scripts in `tests/README.md`.
3. Collect data using `data_collection/README.md`.
4. Train models from `model_training/README.md`.
5. Optimize/deploy with `inference/README.md`.
6. Run fleet workflows from `fleet/fleet_management_app/README.md`.

## Legacy references

Original workflow descriptions are preserved in:

- `setup/jetracer_scripts_README.md`
- `data_collection/training_collection_README.md`
- `data_collection/data_frontend/README.md`
- `setup/pyrealsense2/README.md`

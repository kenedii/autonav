# Inference

This folder contains deployment-time scripts for optimizing trained models and running autonomous control on the car.

## Files

- `run_autonomous_resnet.py`: Main runtime loop that loads the control model and sends PWM control outputs to steering/throttle hardware.
- `trt_optimize.py`: Converts/optimizes PyTorch models for TensorRT so Jetson runtime achieves practical prediction throughput.
- `rknn_buildx86.py`: Utility for building RKNN artifacts on x86 for Rockchip-targeted deployments.
- `requirements.txt`: Python dependencies for inference and optimization flows.

## Why optimization matters

Without TensorRT optimization, Jetson Nano inference throughput can be too slow for responsive control. In this project, TensorRT conversion significantly improved prediction rate and made autonomous control practical.

## Typical deployment flow

1. Train and export/select a model checkpoint from `model_training/`.
2. Run `trt_optimize.py` (Jetson path) or `rknn_buildx86.py` (Rockchip path).
3. Update runtime model paths in `run_autonomous_resnet.py` configuration.
4. Run autonomous script on-car after hardware checks in `tests/`.

## Notes

- Keep architecture names aligned between training and optimization scripts.
- Validate steering/throttle output ranges before high-speed tests.
- Prefer controlled indoor validation before deploying on an open track.

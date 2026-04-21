# Inference

This folder contains deployment-time scripts for optimizing trained models and running autonomous control on the car.

## Files

- `run_autonomous_resnet.py`: Main runtime loop used for the final Jetson demo path. It loads the control model, captures CAM0, runs inference, and sends steering/throttle pulse-width commands to the active motor-control backend.
- `trt_optimize.py`: Converts/optimizes PyTorch models for TensorRT so Jetson runtime achieves practical prediction throughput.
- `rknn_buildx86.py`: Utility for building RKNN artifacts on x86 for Rockchip-targeted deployments.
- `requirements.txt`: Python dependencies for inference and optimization flows.

## Final validated Jetson presentation path

The primary final-presentation inference path is:

- Jetson Nano
- CAM0 primary RGB input
- AutoNav-v2-34 / ResNet34
- TensorRT optimization
- PCA9685 motor control
- `run_autonomous_resnet.py`

Validated command:

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

This is the known-good live demo path for the final technical presentation.

## Why optimization matters

Without TensorRT optimization, Jetson Nano inference throughput can be too slow for responsive control. In this project, TensorRT conversion significantly improved prediction rate and made autonomous control practical.

## Typical deployment flow

1. Train and export/select a model checkpoint from `model_training/`.
2. Run `trt_optimize.py` for the Jetson TensorRT path or `rknn_buildx86.py` for the Rockchip path.
3. Place the selected checkpoint at the expected local model path.
4. Run the validated Jetson live-demo command above for the final presentation path.

## Alternate / legacy paths

- `--controller-backend pca9685`
  - Final validated Jetson presentation path
  - Uses I2C PWM output on the Jetson car

- `--controller-backend pico`
  - Legacy / alternate path
  - Relevant to the Rockchip/Pico platform and older bring-up flows
  - Not the primary final validated Jetson presentation path

## Notes

- Keep architecture names aligned between training and optimization scripts.
- For the final Jetson demo, use CAM0 and the PCA9685 backend.
- The Pico firmware path is still available as an alternate/legacy control backend for the Rockchip platform.
- Use `--serial-port` and `--serial-baud` only when intentionally using the Pico backend.
- Validate steering/throttle output ranges before high-speed tests.
- Prefer controlled indoor validation before deploying on an open track.

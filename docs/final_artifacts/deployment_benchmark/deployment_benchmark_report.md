# Deployment benchmark report

## Status
- Manual procedure only: live deployment benchmark not executed in this pass.

## Environment used for this pass
- Hardware platform: `arm64` host (`Darwin`), not confirmed Jetson Nano
- OS: `macOS-15.6.1-arm64-arm-64bit`
- JetPack: `not detected`
- Python version: `3.11.5`
- Model path: `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth` (present)
- TensorRT model path: `inference/best_model_trt.pth` (missing)

## Validated live path to run on Jetson
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

## Measured deployment metrics in this pass
- FPS: `not measured`
- Average inference time: `not measured`
- Min / max inference time: `not measured`
- Total loop time: `not measured`
- Camera FPS: `not measured`
- Controller backend: `pca9685` (expected validated presentation path, not validated in this pass)

## Hardware validation status
- CAM0 opens: `not validated in this pass`
- Model warms up: `not validated in this pass`
- TensorRT engine loads: `not validated`; local engine file is missing
- PCA9685 backend initializes: `not validated in this pass`
- Manual override path still available: `should be confirmed on the vehicle before demo`

## Why benchmarking was not completed here
- The current environment is a synced development host, not the Jetson Nano presentation target.
- Jetson-specific pieces such as Argus camera access, PCA9685 hardware access, and `tegrastats` are not available here.
- The expected `inference/best_model_trt.pth` file is not present in this repo snapshot.

## Saved frame artifact
- `cam0_runtime_frame.png` was copied from archived local artifact `cam0_model_view_preview.png`.
- Treat it as a representative model-view/runtime image, not as a fresh live capture from this pass.

## Manual procedure to complete on Jetson
1. Confirm the Jetson boots into the known-good environment.
2. Confirm `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth` and `inference/best_model_trt.pth` are present.
3. Start `sudo tegrastats --interval 1000` and save the output to `tegrastats_log.txt`.
4. Run the validated live command with `--debug-timings` and tee stdout/stderr into `timing_log.txt`.
5. Let the system run for 30-60 seconds with a team member on manual override.
6. Stop the run, summarize FPS / inference timings / thermal behavior, and update this folder with the measured values.

## Notes on safety and manual override
- Keep the RC/manual override path active and tested before the benchmark.
- Start with conservative throttle and stop immediately if the vehicle behaves unpredictably.
- Do not present YOLO or SLAM as part of the validated live control loop during this benchmark.

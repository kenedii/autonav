# Final Presentation Status

This document summarizes the final technical-presentation state of the AutoNav repository.

## Validated live feature

- Lane following on Jetson Nano using:
  - CAM0 primary RGB
  - AutoNav-v2-34 / ResNet34
  - TensorRT optimization
  - PCA9685 motor control
  - `inference/run_autonomous_resnet.py`

Known-good command:

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

## Prototype features

- YOLO
  - Prototype only
  - Advisory detection only
  - Not part of the validated live Jetson control loop

- SLAM
  - Experimental RGB-D odometry / replay path
  - Not full production SLAM
  - Not the validated live demo path

- Depth stop
  - Subsystem / prototype only
  - Do not overclaim live validation unless re-tested separately

## Metrics snapshot

| Area | Metric | Value |
|---|---|---|
| Data | raw recorded rows | `9,536` |
| Data | run folders | `17` |
| Model | live model | `AutoNav-v2-34` |
| Inference | observed Jetson runtime | `7.3-7.7 FPS` in prior debug runs |
| Inference | observed TensorRT infer stage | `~91.9 ms` |
| YOLO | smoke-test latency | `~1122.6 ms` CPU-only Jetson smoke test |
| SLAM | room replay result | `x=-0.108 y=8.961 theta=-1.469` |
| Testing | targeted passing tests | AutoNav v2, SLAM core/replay, API, mission, preprocess |

## Demo plan

- Plan A: live lane-following demo
- Plan B: hybrid demo with recorded lane-follow clip plus YOLO/SLAM evidence
- Plan C: code-first fallback with repo walkthrough, screenshots, metrics, and tests

## Limitations to state honestly

- README submodule docs still include older Pico/serial paths and are not all aligned with the final Jetson PCA9685 path
- YOLO is a prototype and not validated in the final live control loop
- SLAM is experimental RGB-D odometry, not full loop-closing SLAM
- Some reported training metrics in the main README are historical reported values and should not be overstated without regenerated artifacts

## Two-member contribution map

- Nicolas Maitland
  - hardware / Jetson integration
  - Jetson-side runtime bring-up and final live-demo validation
  - familiarity with Jetson-side YOLO and SLAM / RGB-D odometry prototype work used during final testing

- Chris Kenedi / `kenedii`
  - model training workflow
  - dashboard / API and broader shared codebase implementation
  - major contributor to the repository foundation

- Shared / collaborative areas
  - data collection
  - deployment support
  - testing / reproducibility / demo support

Use subsystem familiarity during code review rather than claiming one-person ownership of every file.

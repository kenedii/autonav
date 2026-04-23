# Reproducibility summary

## Git state captured at start of this evidence pass

- Branch: `main`
- HEAD: `28099c4efeaf6359a441c812a47e6bbd3b8e1e73`
- Recent commits:
  - `28099c4 Use first names in presentation docs`
  - `75c0a2a Update final presentation documentation`
  - `9db24f0 Expand RealSense RGB-D/IMU capture and SLAM replay`
  - `fb8255b Add AutoNav v2 Jetson runtime support`
  - `e930cb3 Clarify MLP Regressor details in README`
- Working tree at start of pass:
  - `M .gitignore`
  - `?? tests/test_yolo_wrapper.py`

## Known-good platform baseline from repo docs

- Ubuntu `18.04.6`
- JetPack `4.6.1`
- Python `3.6.9`
- Jetson Nano `4GB`
- CAM0 primary RGB
- PCA9685 motor control
- Intel RealSense D435i sidecar / experimental RGB-D/IMU path

## Model weights and runtime entry point

- Final checkpoint path:
  - `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth`
- Validated live demo command:

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

## README / reproducibility checklist

| Topic | Status | Notes |
|---|---|---|
| setup | partial | `setup/README.md` documents the Jetson baseline and build flow, but dependencies remain split across multiple module-specific requirements/setup paths. |
| model weights | partial | Root README documents the expected checkpoint destination and Hugging Face path, but weights are external and not committed. |
| data collection | good | Root README and `data_collection/README.md` cover recorder usage and sensor roles reasonably well. |
| training | partial | `model_training/README.md` exists, but exact final combined dataset artifacts are not checked in. |
| inference | good | Root README and `inference/README.md` now document the validated CAM0 + TensorRT + PCA9685 Jetson path. |
| dashboard | partial | Fleet README distinguishes the host/dashboard role well, but the dashboard is not itself proof of live AI readiness. |
| YOLO | partial | Current docs correctly frame YOLO as prototype/advisory only, but no committed local weights or full benchmark artifact exist. |
| SLAM | partial | Current docs correctly frame SLAM as experimental/replay-oriented, but archived replay data in this workspace is limited to preview-depth runs. |
| tests | partial | Root README lists the passing test commands, but `tests/README.md` is still more generic and does not capture the full validated final test set. |

## Reproducibility gaps to call out honestly

- Root `README.md` still labels commit `9db24f0...` as the known-good final-presentation state, but the current checked-out HEAD is `28099c4...`.
- Model weights are external to Git and must be downloaded separately.
- The validated live command expects `inference/best_model_trt.pth`, which is not guaranteed to be present in every checkout.
- There is no single consolidated environment lockfile or one-shot requirements file for the full repo.
- `tests/test_runtime_split.py` is stale against the current runtime layout.
- `tests/test_data_pipeline_metadata.py` is currently non-green in this host environment because of import-path drift and the local pandas / pyarrow / NumPy compatibility issue.

## Safe final-presentation phrasing

- “The repo is reproducible enough for code review and targeted validation, but still depends on external model weights and the documented Jetson baseline.”
- “The live demo path is clearly documented; some broader or older tests remain stale and are not part of the green final validation set.”

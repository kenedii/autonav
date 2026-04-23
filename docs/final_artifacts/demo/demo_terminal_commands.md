# Demo Terminal Commands

## Optional preflight checks

```bash
git branch --show-current
git rev-parse --short HEAD
ls checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth
ls inference/best_model_trt.pth
```

## Optional API / dashboard support

```bash
python3 fleet/fleet_management_app/client_api/main.py
python3 fleet/fleet_management_app/host_app/server.py
```

## Validated live lane-follow command

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

## Optional higher-throttle retry only if already validated during rehearsal

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
  --throttle 0.30 \
  --no-invert-steering \
  --debug-timings
```

## Evidence files to open quickly if Plan B or C is needed

```bash
open docs/final_artifacts/model_eval/model_eval_report.md
open docs/final_artifacts/testing_reproducibility/testing_summary.md
open docs/final_artifacts/slam/slam_status_report.md
open docs/final_artifacts/deployment_benchmark/deployment_benchmark_report.md
```

If `open` is not appropriate on the presentation machine, navigate to the same files in the editor instead.

# Final Demo Runbook

This runbook uses the current repo evidence and separates the validated live path from prototype or replay-only features.

## Plan A: live lane-following demo

### Purpose

Prove that the validated AutoNav lane-following pipeline runs end to end on the Jetson-side deployment path:

- CAM0 primary RGB
- AutoNav-v2-34 / ResNet34
- TensorRT inference
- PCA9685 motor control
- `inference/run_autonomous_resnet.py`

### Exact commands

Optional host/dashboard preflight:

```bash
python3 fleet/fleet_management_app/client_api/main.py
python3 fleet/fleet_management_app/host_app/server.py
```

Validated live lane-follow command:

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

### Expected outputs

- terminal shows model load and runtime timing lines
- CAM0 opens successfully
- TensorRT engine is found and loaded
- PCA9685 backend initializes
- vehicle follows the lane for at least one controlled segment

### What should be live

- lane-following behavior only
- optional dashboard/status view if it is already working and does not delay the main demo

### What should not be presented as live

- YOLO control behavior
- SLAM-based live navigation
- depth-stop safety behavior unless it is separately re-validated on the actual car before presenting

### Failover triggers

Switch away from Plan A immediately if any of these happen:

- CAM0 fails to open
- `inference/best_model_trt.pth` is missing or the TensorRT path fails
- PCA9685 does not initialize or the car does not respond to control
- lane following is unstable enough that manual override is needed right away

## Plan B: hybrid demo with recorded clips and repo artifacts

### Purpose

Use this if the hardware boots but the car behavior is unreliable or too risky to keep running live.

### Live elements

- Jetson boot and repo state
- optional dashboard/API bring-up
- show the validated command and explain the live path
- open the code and current artifacts live

### Recorded / saved evidence to show

- lane-follow clip from rehearsal or local presentation backup if available
- `docs/final_artifacts/deployment_benchmark/cam0_runtime_frame.png`
- `docs/final_artifacts/data_eda/preprocessing_example.png`
- `docs/final_artifacts/slam/slam_replay_summary.png`
- `docs/final_artifacts/slam/slam_pose_overlay.png`
- `docs/final_artifacts/testing_reproducibility/testing_summary.md`
- `docs/final_artifacts/model_eval/model_eval_report.md`
- `docs/final_artifacts/yolo/yolo_status_report.md`

### Honest framing

- lane following is the validated live feature
- YOLO is a prototype advisory detection path
- SLAM is experimental replay-oriented RGB-D odometry / visual odometry
- the hybrid mode still demonstrates deployment, code structure, and measured evidence without overstating live readiness

### Failover triggers

- live vehicle drifts repeatedly
- audience-facing space is too constrained for safe driving
- dashboard works but the car hardware path is inconsistent

## Plan C: code-first fallback

### Purpose

Use this if the Jetson hardware path is unavailable or the vehicle cannot be safely demonstrated.

### Walkthrough order

1. `data_collection/record_data3.py`
2. `data_collection/preprocess_utils.py`
3. `model_training/train_model_experiments.py`
4. `inference/run_autonomous_resnet.py`
5. `fleet/fleet_management_app/client_api/models.py`
6. `fleet/fleet_management_app/client_api/slam.py`
7. `fleet/fleet_management_app/client_api/main.py`

### Artifact set to show

- `docs/final_artifacts/data_eda/*`
- `docs/final_artifacts/model_eval/*`
- `docs/final_artifacts/deployment_benchmark/*`
- `docs/final_artifacts/yolo/*`
- `docs/final_artifacts/slam/*`
- `docs/final_artifacts/testing_reproducibility/*`
- `docs/final_artifacts/team_ownership/*`

### Expected outcome

- satisfy the technical rubric with concrete code paths, plots, metrics, and test evidence
- keep the presentation honest by separating validated deployment from prototype features

### Failover trigger

- any hardware issue that makes a live run unpredictable or impossible

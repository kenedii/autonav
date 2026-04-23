# Demo Checklist

## Hardware and power

- battery charged for the car
- Jetson power source stable
- RC/manual controller available and paired if applicable

## Jetson and software bring-up

- Jetson boots into the known-good environment
- repo is on the expected branch / commit for the presentation
- no stale autonomous process is already running

## Camera and model path

- CAM0 opens successfully
- forward-facing view is correct
- `checkpoints/AutoNav-v2/AutoNav-v2-34/AutoNav-v2-34.pth` exists
- `inference/best_model_trt.pth` exists on the presentation Jetson

## Control path

- PCA9685/control path responds before the autonomous run
- manual override path is confirmed and assigned to one team member

## Dashboard and support tooling

- `fleet/fleet_management_app/client_api/main.py` starts if you plan to show API status
- `fleet/fleet_management_app/host_app/server.py` starts if you plan to show the dashboard
- dashboard is treated as support tooling, not as the proof of autonomy

## Prototype evidence checks

- YOLO evidence package is available:
  - `docs/final_artifacts/yolo/yolo_status_report.md`
  - `docs/final_artifacts/yolo/yolo_smoke_result.md`
- SLAM replay evidence is available:
  - `docs/final_artifacts/slam/slam_status_report.md`
  - `docs/final_artifacts/slam/slam_replay_summary.png`
  - `docs/final_artifacts/slam/slam_pose_overlay.png`

## Testing and reproducibility checks

- `docs/final_artifacts/testing_reproducibility/testing_summary.md` is open or easy to reach
- `docs/final_artifacts/testing_reproducibility/reproducibility_summary.md` is open or easy to reach

## Final presentation backups

- lane-follow rehearsal clip available locally if Plan B is needed
- dashboard screenshot available if the live dashboard does not cooperate
- Jetson timing screenshot available if live timing capture is not possible
- pytest output screenshot or `testing_summary.md` available

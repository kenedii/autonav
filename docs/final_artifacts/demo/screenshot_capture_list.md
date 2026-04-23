# Screenshot Capture List

Capture these before the final presentation so Plan B and Plan C are fully supported.

## Lane-follow evidence

- live lane-follow clip from the actual Jetson presentation setup
- one still frame showing the car on track during the live run

## Dashboard evidence

- dashboard home/status view while the car or API is online
- any useful deployment or log view that proves the dashboard is connected

## CAM0 model-view evidence

- fresh model-view screenshot from the Jetson runtime if possible
- fallback currently available:
  - `docs/final_artifacts/deployment_benchmark/cam0_runtime_frame.png`
  - `docs/final_artifacts/data_eda/preprocessing_example.png`

## YOLO evidence

- annotated YOLO frame if a local YOLO weight file is available and a smoke test can be rerun
- current status artifact already available:
  - `docs/final_artifacts/yolo/yolo_status_report.md`
- current gap:
  - no committed `yolo_annotated_example.png` exists yet because local YOLO weights were missing

## SLAM evidence

- trajectory screenshot:
  - `docs/final_artifacts/slam/slam_replay_summary.png`
- pose-overlay screenshot:
  - `docs/final_artifacts/slam/slam_pose_overlay.png`

## Testing evidence

- pytest terminal screenshot showing the passing targeted test set
- fallback currently available:
  - `docs/final_artifacts/testing_reproducibility/testing_summary.md`

## Deployment timing evidence

- Jetson terminal screenshot from `--debug-timings`
- `tegrastats` capture from the actual Jetson presentation run
- fallback currently available:
  - `docs/final_artifacts/deployment_benchmark/deployment_benchmark_report.md`
- current gap:
  - no fresh Jetson-captured timing artifact was generated in this workspace because this host is not the Jetson

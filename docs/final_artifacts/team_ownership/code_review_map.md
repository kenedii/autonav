# Code Review Map

This map is intended for the final instructor code review. "Owner / presenter" means the team member who should open the file first and lead the explanation.

## Data collection

- Files
  - `data_collection/record_data3.py`
  - `data_collection/realsense_full.py`
  - `data_collection/net_controller_client.py`
- Owner / presenter
  - Nicolas Maitland and Chris Kenedi / `kenedii`
- What they should explain
  - how manual steering/throttle labels are captured
  - how run folders and `dataset.csv` rows are written
  - how CAM0, CAM1, IR, and depth data are recorded
  - how newer RealSense RGB-D / IMU capture fits into the later prototype work
- Likely questions
  - How were labels collected?
  - Which camera is the primary training camera?
  - What sensor fields are missing in older runs?

## Preprocessing

- Files
  - `data_collection/preprocess_utils.py`
  - `data_collection/data_frontend/dataset_csv_creator.py`
- Owner / presenter
  - Nicolas Maitland
- What they should explain
  - the `cam0_fisheye_v1` path used for the final live model
  - how preprocessing consistency is preserved between training and inference
  - how dataset CSVs are prepared for model training
- Likely questions
  - Why CAM0 fisheye instead of another camera path?
  - How is the model input resized to `160x120`?
  - How do you know inference preprocessing matches training preprocessing?

## Model training

- Files
  - `model_training/train_model_experiments.py`
  - `model_training/train_model_resnet.py`
  - `model_training/dataset_loader.py`
- Owner / presenter
  - Chris Kenedi / `kenedii`
- What they should explain
  - experiment-based training flow
  - train / validation / test split logic
  - normalized steering and throttle targets
  - why AutoNav-v2-34 is the final selected live model
- Likely questions
  - Why ResNet34?
  - What models were tried before the final choice?
  - How do you discuss overfitting or underfitting honestly?

## Inference / deployment

- Files
  - `inference/run_autonomous_resnet.py`
  - `inference/trt_optimize.py`
  - `fleet/fleet_management_app/client_api/models.py`
- Owner / presenter
  - Nicolas Maitland
- What they should explain
  - how the Jetson runtime loads the AutoNav-v2-34 checkpoint
  - TensorRT optimization and the validated live command
  - CAM0 capture, model inference, and PCA9685 motor control
  - where manual override and safety boundaries sit during the demo
- Likely questions
  - What exactly runs on the Jetson?
  - What is the controller backend?
  - What timing or FPS evidence do you have?

## YOLO

- Files
  - `fleet/fleet_management_app/client_api/models.py`
  - `fleet/fleet_management_app/client_api/car.py`
  - `fleet/fleet_management_app/client_api/main.py`
  - `docs/final_artifacts/yolo/yolo_status_report.md`
- Owner / presenter
  - Nicolas Maitland
- What they should explain
  - the `ObjectDetector` wrapper
  - advisory detection output format
  - where detections enter car state and API responses
  - why YOLO is still prototype-only and not part of the live control loop
- Likely questions
  - Does YOLO affect steering or braking?
  - What thresholds are configured?
  - What benchmark evidence do you actually have?

## SLAM / RGB-D odometry

- Files
  - `fleet/fleet_management_app/client_api/slam.py`
  - `tests/test_slam.py`
  - `tests/test_slam_core.py`
  - `tests/test_slam_replay.py`
- Owner / presenter
  - Nicolas Maitland
- What they should explain
  - the optical-flow / RGB-D odometry pipeline
  - pose state output: `x`, `y`, `theta`, trajectory, and motion diagnostics
  - replay evidence and current limitations
  - why this is experimental localization rather than production SLAM
- Likely questions
  - Is this full SLAM or odometry?
  - What happened in the replayed run?
  - Why is live IMU support not a final-demo claim?

## Dashboard / API

- Files
  - `fleet/fleet_management_app/client_api/main.py`
  - `fleet/fleet_management_app/host_app/server.py`
  - `fleet/fleet_management_app/host_app/static/app.js`
  - `tests/test_server.py`
  - `tests/test_client.py`
- Owner / presenter
  - Chris Kenedi / `kenedii`
- What they should explain
  - host/client split
  - configure / start / stop / status flow
  - deployment support, logs, and monitoring role of the dashboard
  - why the dashboard is not the main proof of YOLO or SLAM production readiness
- Likely questions
  - What does the dashboard actually control?
  - Which API tests are passing?
  - Does the dashboard visualize detections or SLAM deeply?

## Tests / reproducibility

- Files
  - `docs/final_artifacts/testing_reproducibility/testing_summary.md`
  - `docs/final_artifacts/testing_reproducibility/reproducibility_summary.md`
  - `tests/test_mission.py`
  - `tests/test_preprocess_utils.py`
- Owner / presenter
  - Nicolas Maitland and Chris Kenedi / `kenedii`
- What they should explain
  - which targeted tests are green
  - which tests are stale and should not be overstated
  - the known-good commit and final demo command
  - how the repo can be reproduced from GitHub with external weights
- Likely questions
  - What is actually verified?
  - Why is `tests/test_runtime_split.py` not part of the passing set?
  - What still depends on external hardware or weights?

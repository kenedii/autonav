# Tests

This folder contains validation, calibration, and regression scripts used to verify camera, controller, and runtime behavior.

## Files

- `camera_cv2.py`: Basic OpenCV camera capture test.
- `take_photo_test.py`: Captures still images to validate camera pipeline and storage.
- `xbox_calibrate.py`: Calibrates an Xbox controller on the Jetson.
- `car_controller_test.py`: Manual driving test after controller calibration.
- `test_data_pipeline_metadata.py`: Data-pipeline metadata checks.
- `test_preprocess_utils.py`: Preprocessing utility tests.
- `test_client.py`, `test_mission.py`, `test_runtime_split.py`, `test_slam.py`: Client-side and SLAM-related tests.
- `test_server.py`: Host application server-side tests.

## Typical usage

Run targeted scripts directly while iterating on hardware and integration:

```bash
python tests/xbox_calibrate.py
python tests/car_controller_test.py
python tests/test_server.py
```

For hardware tests, ensure cameras, controller, and motor control hardware are connected before execution.

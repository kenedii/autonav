# Slide-ready YOLO bullets

- YOLO exists in the fleet runtime as a lightweight `ObjectDetector` wrapper around Ultralytics models.
- The repo-normalized output is advisory only: class ID, bounding box, confidence, and optional depth-derived distance.
- In the current product path, YOLO does **not** steer, brake, or override the lane-follow model.
- Dashboard config includes a YOLO model name, but the current deploy UI does not enable detection by default and does not render a dedicated detections panel.
- This audit pass verified the code path and added a smoke-test script, but a real benchmark still requires a local YOLO weight file.
- Final presentation wording should be: prototype perception feature with code and wrapper evidence, not a live-validated safety system.

# YOLO prototype status report

## Where YOLO exists in the repo
- `fleet/fleet_management_app/client_api/models.py`
  - defines `ObjectDetector`
  - loads Ultralytics `YOLO(...)` if the dependency is installed and the weight path exists
- `fleet/fleet_management_app/client_api/car.py`
  - instantiates `ObjectDetector` when `'detection'` is included in `action_loop`
  - runs `detect(frame_color)` inside the action loop
  - optionally appends `distance` using the depth frame
  - stores detections in `state["detections"]`
- `fleet/fleet_management_app/client_api/main.py`
  - includes `detection_model` in `ClientConfig`
  - exposes state through `/status`
- `fleet/fleet_management_app/host_app/static/app.js`
  - deploy UI includes `detection_model: "yolov8n.pt"`
  - deploy UI currently sends `action_loop: ["control", "api"]`, so detection is not enabled by default from the dashboard
- `fleet/fleet_management_app/host_app/server.py`
  - contains virtual/mock `yolo_version` status and fake log lines only

## Model and weights
- Model name used by current configs: `yolov8n.pt`
- Weight source: not specified in repo docs
- Weight file committed to repo: no

## Inference configuration known from code
- Input size: not specified in repo code; raw frame is passed directly to Ultralytics and resize behavior is delegated to the library/model defaults
- Confidence threshold: not specified in repo code
- NMS threshold: not specified in repo code
- Classes available: not encoded in repo; detections are returned as integer class IDs and depend on the loaded YOLO weights

## Output format
The repo-normalized detection output is:

```python
[
  {
    "class": int,
    "bbox": [x1, y1, x2, y2],
    "conf": float
  }
]
```

When a depth frame is available, `car.py` may append:

```python
{
  "distance": float
}
```

## Dashboard / control integration
- Dashboard display: no dedicated YOLO visualization was found in `host_app/static/app.js`
- API visibility: detections are accessible indirectly through `state["detections"]` and `/status`
- Control use: none
- Steering / throttle / braking use: none

## Runtime maturity
- Status: prototype / advisory only
- Suitable for: code review, smoke-test evidence, future perception integration
- Not suitable to claim: validated live driving behavior, safety intervention, production obstacle avoidance, or dashboard-ready perception UX

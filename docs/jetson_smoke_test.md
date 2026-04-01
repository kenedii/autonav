# Jetson Smoke-Test Checklist

Use this checklist on the Jetson and vehicle after deploying the CAM0-primary split. This checklist is manual by design. It does not claim hardware success in the development environment.

## 1. Start the host dashboard

Command:

```bash
python host_app/server.py
```

Expected:
- the host dashboard loads
- the main preview card is labeled `CAM0 Forward Preview`
- the `Sensor Roles` panel shows CAM0, RealSense, and CAM1 as separate roles

Pass criteria:
- no dashboard JavaScript errors
- the validation strip is visible with `Tag Detector`, `Control Model`, `Depth`, and `Stop Reason`

## 2. Start the Jetson client

Command:

```bash
python client_api/main.py --host <HOST_IP> --name Jetson
```

Expected:
- the client connects to the host
- the host shows the vehicle online
- the client continues using the current websocket and `/ws/video` flow

Pass criteria:
- the host shows a connected car entry
- no connection/authentication errors repeat in the client log

## 3. Apply the checked-in CAM0-primary config

Command:

```bash
curl -s -X POST http://<JETSON_IP>:8000/configure \
  -H "X-Api-Key: changeme" \
  -H "Content-Type: application/json" \
  --data @docs/examples/client_config_cam0_primary.json
```

Expected:
- CAM0 is configured as `primary_rgb`
- RealSense is configured as `sidecar_depth_imu`
- CAM1 is configured as `rear_preview` and disabled by default
- `preprocess_profile == "cam0_fisheye_v1"`

Pass criteria:
- the configure call returns `{"status":"configured"}`
- a follow-up `/status` shows `state.sensors.primary_rgb.source == "cam0"`
- a follow-up `/status` shows `state.sensors.depth.source == "realsense"`
- a follow-up `/status` shows `state.sensors.imu.source == "realsense"`

## 4. Start autonomy

Command:

```bash
curl -s -X POST http://<JETSON_IP>:8000/start -H "X-Api-Key: changeme"
```

Expected:
- the car enters the current AutoNav/control loop
- the validation strip updates for tag detector, control model, depth, and stop reason

Pass criteria:
- the start call returns `{"status":"started"}`
- `state.mission.state` becomes active for the configured mission

## 5. Inspect runtime state

Command:

```bash
curl -s http://<JETSON_IP>:8000/status -H "X-Api-Key: changeme"
```

Expected fields:
- `state.sensors.primary_rgb.source == "cam0"`
- `state.sensors.primary_rgb.used_for` includes `lane_following`, `apriltag`, `forward_preview`
- `state.sensors.primary_rgb.frame_age_ms`
- `state.sensors.depth.source == "realsense"`
- `state.sensors.depth.status`
- `state.sensors.depth.frame_age_ms`
- `state.sensors.imu.source == "realsense"`
- `state.sensors.imu.status`
- `state.sensors.rear.status`
- `state.mission.stop_reason`
- `state.mission.tag_detector_status`
- `state.mission.control_model_status`
- `state.mission.depth_status`

Pass criteria:
- all fields above are present and JSON-safe
- the dashboard matches the `/status` role split

## 6. CAM0 forward preview test

Trigger:
- leave CAM0 connected and healthy
- select the vehicle in the host dashboard

Observe:
- the preview updates live
- the preview heading remains `CAM0 Forward Preview`
- no UI element suggests RealSense RGB or CAM1 is the forward camera

Pass criteria:
- forward preview updates while CAM0 is present
- `state.sensors.primary_rgb.status` is `configured` or `available`, then settles to `available`

## 7. CAM0 lane-follow runtime test

Trigger:
- place the car on the lane-follow track
- keep CAM0 and RealSense connected
- start the control loop as above

Observe:
- steering responds to the CAM0 view
- CAM1 remains irrelevant to forward control

Pass criteria:
- `state.sensors.primary_rgb.used_for` continues to show forward-control ownership
- the car can follow the lane with CAM0 connected

## 8. CAM0 AprilTag checkpoint detection test

Trigger:
- present the configured AprilTags to CAM0 during a mission run

Observe:
- mission checkpoint progress updates
- `state.mission.tag_detector_status` stays visible on the dashboard
- `state.mission.last_tag_id` updates when a tag is seen

Pass criteria:
- checkpoint tags are detected from CAM0
- tag detection status is visible even if the detector is degraded or unavailable

## 9. RealSense depth obstacle-stop test

Trigger:
- with depth-stop enabled, place an obstacle in the configured front ROI

Observe:
- the car performs a safe stop
- the dashboard validation strip updates the depth/stop fields

Pass criteria:
- `state.mission.stop_reason == "obstacle"`
- throttle is neutralized
- `state.mission.depth_status` remains visible

## 10. RealSense IMU visibility/state test

Command:

```bash
curl -s http://<JETSON_IP>:8000/status -H "X-Api-Key: changeme"
```

Observe:
- `state.location.imu` contains accel/gyro when IMU data is present
- `state.sensors.imu.status` is visible separately from depth

Pass criteria:
- the operator can tell whether IMU is available without digging through logs

## 11. Missing CAM0 fallback test

Trigger:
- while autonomy is running, disconnect CAM0 or otherwise force CAM0 frame loss

Command to observe:

```bash
curl -s http://<JETSON_IP>:8000/status -H "X-Api-Key: changeme"
```

Expected:
- safe stop / neutral output
- an explicit CAM0-related stop reason
- no takeover by CAM1 or RealSense RGB

Pass criteria:
- `state.mission.stop_reason == "primary_rgb_unavailable"`
- `state.sensors.primary_rgb.status == "unavailable"`
- the logs contain `Primary RGB camera unavailable; neutralizing outputs.`

## 12. Missing depth fallback test

Trigger:
- keep CAM0 connected
- disconnect RealSense depth or otherwise stop depth frames while depth-stop is enabled

Command to observe:

```bash
curl -s http://<JETSON_IP>:8000/status -H "X-Api-Key: changeme"
```

Expected:
- the car stops after the short grace window
- the stop reason is depth-related

Pass criteria:
- `state.mission.stop_reason == "depth_unavailable"`
- `state.sensors.depth.status == "unavailable"`
- the logs contain `RealSense depth missing beyond grace window; stopping autonomy.`

## 13. Missing IMU visibility without forced stop

Trigger:
- keep CAM0 and RealSense depth healthy
- disable or disconnect only the IMU stream if possible

Command to observe:

```bash
curl -s http://<JETSON_IP>:8000/status -H "X-Api-Key: changeme"
```

Expected:
- IMU status becomes unavailable
- forward control does not stop solely because IMU is missing

Pass criteria:
- `state.sensors.imu.status == "unavailable"`
- `state.location.imu` becomes `null` or empty rather than staying stale
- `state.mission.stop_reason` does not become IMU-related when CAM0 and depth are healthy

## 14. Optional CAM1 rear-preview test

Trigger:
- edit `docs/examples/client_config_cam0_primary.json` locally to set the `rear_preview` camera `enabled` flag to `true`
- re-run the configure call from step 3

Observe:
- rear preview status becomes visible as enabled
- rear failures remain warning-only

Pass criteria:
- CAM1 still does not affect lane following, AprilTags, or forward preview
- a missing rear camera does not stop the car while CAM0 is healthy

## 15. Manual override / stop-resume check

Commands:

```bash
curl -s -X POST http://<JETSON_IP>:8000/stop -H "X-Api-Key: changeme"
curl -s -X POST http://<JETSON_IP>:8000/start -H "X-Api-Key: changeme"
curl -s -X POST http://<JETSON_IP>:8000/pause -H "X-Api-Key: changeme" -H "Content-Type: application/json" --data '{"duration":2}'
curl -s -X POST http://<JETSON_IP>:8000/resume -H "X-Api-Key: changeme"
```

Expected:
- operator stop and restart still work
- pause and resume still work
- current stop reason and sensor health remain visible on the dashboard

Pass criteria:
- stop/start/pause/resume all return success responses
- the operator can stop the vehicle safely without changing config

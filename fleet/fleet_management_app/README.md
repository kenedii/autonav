# Fleet Management App

This folder contains the two components used for multi-car management:

- `client_api/`: Runs on each car and exposes control/status APIs.
- `host_app/`: Runs centrally and provides fleet management APIs + frontend UI.

## Architecture

Each car runs `client_api/main.py` as an on-car service. The host connects to each car over HTTP/WebSocket for:

- status streaming
- command dispatch (start/stop/pause/resume)
- settings updates (throttle mode, fixed throttle)
- remote optimization/deployment flows

## Runtime paths in this repo

There are multiple runtime paths represented here. For the final presentation, keep them distinct:

- Jetson car path
  - Final validated live-demo path uses CAM0 + AutoNav-v2-34 + TensorRT + PCA9685 motor control
  - The strongest live demo remains the direct Jetson runtime in `inference/run_autonomous_resnet.py`

- Rockchip / Pico path
  - Alternate platform path using a Raspberry Pi Pico and L298N motor driver
  - Still represented in the fleet/client code and older docs
  - Not the primary final validated presentation path

- Dashboard / host path
  - Used for deployment, status, logs, and operator monitoring
  - Useful for code review and operational demos
  - Not by itself proof that every AI subsystem is production-ready

- Prototype perception/localization paths
  - YOLO exists as a prototype detection path
  - SLAM exists as an experimental RGB-D odometry/replay path
  - Neither should be presented as the validated final live-driving path

Control path on the car side depends on platform:

- Jetson final presentation path:
  - runtime performs inference on the Jetson
  - control output is sent through the Jetson motor-control path used by the validated live demo
  - for the final reviewed demo, that path is PCA9685-based and lives primarily in `inference/run_autonomous_resnet.py`

- Rockchip / alternate fleet path:
  - client runtime can forward steering/throttle pulse-width targets over serial/USB to a Raspberry Pi Pico
  - Pico firmware generates hardware PWM and drives the L298N module

## 1) Setup the client on each car (Jetson)

1. Copy the `client_api/` folder to the Jetson.
2. Create/activate a Python environment.
3. Install dependencies required by FastAPI, model runtime, and hardware stack.
4. Configure the runtime password and model settings through the `/configure` endpoint.
5. Start the API service on the car.

Example start command:

```bash
python fleet/fleet_management_app/client_api/main.py
```

Default API port is `8000` unless overridden.

This client API is appropriate for:

- status and control endpoints
- host/dashboard integration
- prototype YOLO and SLAM state reporting

It is not the only runtime entry point in the repo. The final validated live lane-following demo path is still the direct Jetson runtime in `inference/run_autonomous_resnet.py`.

## 2) Setup fleet management host frontend

1. Go to `host_app/`.
2. Install Python dependencies for the host server.
3. If using the static UI assets, ensure the server can serve `host_app/static/`.
4. Start the host server.

Example:

```bash
python fleet/fleet_management_app/host_app/server.py
```

After startup, open the host URL in a browser to access the fleet dashboard.

## Feature maturity inside the fleet app

- Lane following / control
  - Fleet client can host the control model path
  - The strongest validated live demo remains the direct Jetson runtime outside the fleet app

- YOLO
  - Prototype / advisory only
  - Detection results can be exposed through runtime state and APIs
  - Do not present this as validated live production behavior

- SLAM
  - Experimental RGB-D odometry / replay-oriented support
  - Useful for code review, replay, and future localization discussion
  - Do not present this as full production SLAM or the validated live demo path

- Dashboard
  - Good for deployment, status, logs, and monitoring
  - Not the main proof that YOLO or SLAM are production-ready

## Security and auth

- Both host and client use API-key style authentication via `X-Api-Key`.
- Keep per-car passwords aligned between host configuration and each client.
- Optional Fernet encryption is supported when `cryptography` is installed.

## File guide

- `client_api/main.py`: Car-side FastAPI service and runtime control endpoints.
- `client_api/car.py`: Car control and runtime integration.
- `client_api/mission.py`: Mission/task logic utilities.
- `client_api/slam.py`: SLAM integration helpers.
- `client_api/tag_detector.py`: Tag detection utilities.
- `host_app/server.py`: Fleet host API and websocket manager.
- `host_app/host.py`: Host-side helper/agent utilities for interacting with clients.
- `host_app/static/index.html`: Fleet dashboard frontend.
- `host_app/static/app.js`: Frontend behavior.
- `host_app/static/style.css`: Dashboard styling.

## Recommended bring-up order

1. Bring up one car client and verify `/status`.
2. Start the host server and register that car.
3. Confirm websocket status updates and command round-trips.
4. Scale to additional cars.

For final-presentation purposes, use the fleet app to demonstrate:

- deployment/status/logging workflow
- operator monitoring
- API surface for the car

Use the direct Jetson runtime for the strongest live lane-following proof.

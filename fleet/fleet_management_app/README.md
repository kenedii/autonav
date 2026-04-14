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

Control path on the car side:

- Rockchip/Jetson runtime performs inference.
- Client API forwards steering/throttle pulse-width targets over serial/USB to a Raspberry Pi Pico.
- Pico firmware generates hardware PWM and drives the L298N module, which controls the two DC motors.

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

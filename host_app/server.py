from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Deque
from pathlib import Path
import requests
import uvicorn
import json
import os
import asyncio
import uuid
import random
import hmac
import hashlib
import base64
import secrets
import time
from collections import deque
import websockets as _ws_client  # outbound WS connections to the car
from fastapi.middleware.cors import CORSMiddleware

# Maximum log entries kept per car in server memory
_MAX_LOG_ENTRIES = 500

# ── Encryption helpers (mirrors client) ─────────────────────────────────────
try:
    from cryptography.fernet import Fernet, InvalidToken
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False

def _derive_fernet_key(password: str) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(digest)

def _make_fernet(password: str):
    if not _FERNET_AVAILABLE or not password:
        return None
    return Fernet(_derive_fernet_key(password))

def _encrypt_payload(data: dict, fernet) -> str:
    if fernet is None:
        return json.dumps(data)
    return fernet.encrypt(json.dumps(data).encode()).decode()

def _decrypt_payload(token: str, fernet) -> dict:
    if fernet is None:
        return json.loads(token)
    return json.loads(fernet.decrypt(token.encode()))

def _hmac_challenge(password: str, nonce: str) -> str:
    return hmac.new(password.encode(), nonce.encode(), hashlib.sha256).hexdigest()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class NewClient(BaseModel):
    name: str
    ip: str
    port: int = 8000
    password: str = "changeme"  # Must match the password configured on the car

class ClientSettings(BaseModel):
    throttle_mode: Optional[str] = None
    fixed_throttle_value: Optional[float] = None

class DeployConfig(BaseModel):
    config: Dict[str, Any]

# --- In-Memory Database ---
class CarDb:
    def __init__(self):
        self.cars: Dict[str, Dict] = {}
        # Per-car ring buffer of log entries { timestamp, level, message }
        self.logs: Dict[str, Deque] = {}
        # Structure:
        # { "ip:port": { "name", "ip", "port", "password", "status", "details", "fernet", ... } }

db = CarDb()

def _append_log(car_id: str, entries: List[Dict]):
    """Append new log entries to the per-car ring buffer."""
    if car_id not in db.logs:
        db.logs[car_id] = deque(maxlen=_MAX_LOG_ENTRIES)
    buf = db.logs[car_id]
    for entry in entries:
        # Normalise: make sure every entry has the expected fields
        buf.append({
            "timestamp": entry.get("timestamp", time.time()),
            "level":     entry.get("level", "INFO").upper(),
            "message":   str(entry.get("message", "")),
        })

# Convenience: look up the Fernet cipher for a car (or None)
def _get_fernet(car_id: str):
    car = db.cars.get(car_id)
    if not car:
        return None
    return car.get("fernet")

# --- Helpers ---
def get_geo_info(ip: str):
    # Skip private IPs
    if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("127."):
        return {"country": "Local Network", "city": "N/A", "lat": 0, "lon": 0}
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=1)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        # car_id -> {"ws": WebSocket, "fernet": Fernet|None}
        self.active_connections: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, car_id: str, fernet=None):
        """Accept the connection and store it (websocket must already be accepted)."""
        self.active_connections[car_id] = {"ws": websocket, "fernet": fernet}

    def disconnect(self, car_id: str):
        if car_id in self.active_connections:
            del self.active_connections[car_id]

    async def send_command(self, car_id: str, command: dict):
        """Send an encrypted command to a car."""
        entry = self.active_connections.get(car_id)
        if not entry:
            return False
        payload = _encrypt_payload(command, entry["fernet"])
        await entry["ws"].send_text(payload)
        return True

manager = ConnectionManager()

@app.websocket("/ws/car/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
    """
    Handles inbound WebSocket connections from JetRacer clients.

    Auth + Encryption protocol:
      1. Server sends  {"type": "challenge", "nonce": "<hex>"}
      2. Client must reply {"type": "auth", "response": HMAC(password, nonce)}
      3. Server replies {"type": "auth_ok"} and enters normal operation,
         or closes with 4401 on failure.
      4. All subsequent messages are Fernet-encrypted JSON.
    """
    ip = websocket.client.host
    car_id = f"{ip}:WS"

    # Auto-register if not seen before (password defaults to "changeme" until
    # the car is explicitly added via /api/cars with a password).
    if car_id not in db.cars:
        db.cars[car_id] = {
            "id": car_id,
            "name": client_name,
            "ip": ip,
            "port": 0,
            "password": "changeme",
            "status": "online",
            "details": {},
            "geo": get_geo_info(ip),
            "type": "websocket",
            "fernet": _make_fernet("changeme"),
        }

    car_record = db.cars[car_id]
    password = car_record.get("password", "changeme")
    fernet = _make_fernet(password)

    await websocket.accept()

    # ── Step 1: send challenge ───────────────────────────────────────────────
    nonce = secrets.token_hex(32)
    await websocket.send_text(json.dumps({"type": "challenge", "nonce": nonce}))

    # ── Step 2: verify HMAC response ────────────────────────────────────────
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        auth_msg = json.loads(raw)
    except (asyncio.TimeoutError, json.JSONDecodeError):
        await websocket.close(code=4401)
        return

    expected = _hmac_challenge(password, nonce)
    provided  = auth_msg.get("response", "")
    if not hmac.compare_digest(expected, provided):
        await websocket.send_text(json.dumps({"type": "auth_failed"}))
        await websocket.close(code=4401)
        print(f"[WS] Auth FAILED for {client_name} ({ip})")
        return

    # ── Step 3: confirm auth ─────────────────────────────────────────────────
    await websocket.send_text(json.dumps({"type": "auth_ok"}))
    print(f"[WS] Auth OK for {client_name} ({ip})")

    # Update fernet now that we confirmed the password is correct
    car_record["fernet"] = fernet
    await manager.connect(websocket, car_id, fernet)

    # Record connection event in logs
    _append_log(car_id, [{"timestamp": time.time(), "level": "INFO",
                           "message": f"Client '{client_name}' connected from {ip}."}])

    # ── Step 4: normal operation (receive encrypted state) ───────────────────
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = _decrypt_payload(raw, fernet)
            except Exception:
                print(f"[WS] Failed to decrypt message from {client_name} – skipping.")
                continue
            if car_id in db.cars:
                # Separate out log entries before storing the rest of the state
                log_entries = data.pop("logs", [])
                if log_entries:
                    _append_log(car_id, log_entries)

                db.cars[car_id]["details"] = data
                db.cars[car_id]["status"] = "online" if data.get("running") else "stopped"
    except WebSocketDisconnect:
        manager.disconnect(car_id)
        if car_id in db.cars:
            db.cars[car_id]["status"] = "offline"
            _append_log(car_id, [{"timestamp": time.time(), "level": "WARN",
                                   "message": f"Client '{client_name}' disconnected."}])

# ── Video proxy WebSocket ────────────────────────────────────────────────────
@app.websocket("/ws/video/{car_id}")
async def video_proxy_endpoint(websocket: WebSocket, car_id: str):
    """
    Browser connects here; we open an outbound WS to the car's /ws/video
    endpoint, authenticate, and relay encrypted JPEG frames to the browser
    (decrypted server-side, sent as raw binary to the browser).
    """
    await websocket.accept()

    # Look up car
    # car_id in URL is URL-encoded "ip:port" or "ip:WS"
    car_record = db.cars.get(car_id)
    if not car_record:
        await websocket.close(code=4404)
        return

    car_ip   = car_record["ip"]
    car_port = car_record.get("port") or 8000
    password = car_record.get("password", "changeme")
    fernet   = _make_fernet(password)

    car_ws_uri = f"ws://{car_ip}:{car_port}/ws/video"

    try:
        async with _ws_client.connect(car_ws_uri) as car_ws:
            # Send auth to the car
            await car_ws.send(json.dumps({"auth": password}))
            ack_raw = await asyncio.wait_for(car_ws.recv(), timeout=10.0)
            ack = json.loads(ack_raw)
            if ack.get("status") != "ok":
                await websocket.send_text(json.dumps({"error": "car_auth_failed"}))
                await websocket.close(code=4401)
                return

            # Relay frames: decrypt from car, forward raw JPEG to browser
            async for frame_data in car_ws:
                try:
                    if fernet is not None and isinstance(frame_data, bytes):
                        jpeg = fernet.decrypt(frame_data)
                    else:
                        jpeg = frame_data
                    await websocket.send_bytes(jpeg)
                except WebSocketDisconnect:
                    break
                except Exception:
                    continue
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close(code=1011)
        except Exception:
            pass

# --- Hybrid Proxy Helpers ---
async def send_car_command(car_id: str, command: str, payload: dict = None):
    if car_id not in db.cars: raise HTTPException(404, "Car not found")
    car = db.cars[car_id]
    payload = dict(payload or {})
    if command == "configure" and "password" not in payload:
        payload["password"] = car.get("password", "changeme")

    if car.get("type") == "virtual":
        return {"status": "sent_via_virtual"}

    # WebSocket Client – send encrypted command
    if car.get("type") == "websocket":
        cmd_payload = {"command": command}
        cmd_payload.update(payload)
        success = await manager.send_command(car_id, cmd_payload)
        if not success: raise HTTPException(503, "Car disconnected")
        return {"status": "sent_via_ws"}

    # Legacy HTTP Client – attach X-Api-Key header
    try:
        url = f"http://{car['ip']}:{car['port']}/{command}"
        headers = {"X-Api-Key": car.get("password", "changeme")}
        if payload:
            requests.post(url, json=payload, headers=headers, timeout=2)
        else:
            requests.post(url, headers=headers, timeout=2)
        return {"status": "sent_via_http"}
    except Exception as e:
        raise HTTPException(500, f"HTTP Error: {e}")

# --- Endpoints ---

def _safe_car(record: dict) -> dict:
    """Return a JSON-serialisable copy of a car record with secrets redacted."""
    return _redact_sensitive(record)


def _redact_sensitive(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in ("fernet", "password"):
                continue
            redacted[key] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _test_client_role_cameras():
    return [
        {
            "role": "primary_rgb",
            "type": "csi",
            "sensor_id": 0,
            "width": 640,
            "height": 480,
            "fps": 15,
            "flip_method": 2,
            "enabled": True,
        },
        {
            "role": "sidecar_depth_imu",
            "type": "realsense",
            "width": 640,
            "height": 480,
            "fps": 15,
            "enabled": True,
        },
        {
            "role": "rear_preview",
            "type": "csi",
            "sensor_id": 1,
            "width": 640,
            "height": 480,
            "fps": 15,
            "flip_method": 2,
            "enabled": False,
        },
    ]


def _test_client_sensor_state():
    return {
        "forward_preview_role": "primary_rgb",
        "depth_aligned_to_primary": False,
        "primary_rgb": {
            "role": "primary_rgb",
            "type": "csi",
            "label": "CAM0 Primary RGB",
            "configured": True,
            "enabled": True,
            "present": True,
            "healthy": True,
            "status": "available",
            "frame_available": True,
            "depth_available": False,
            "imu_available": False,
            "source": "cam0",
            "used_for": ["lane_following", "apriltag", "forward_preview"],
            "sensor_id": 0,
            "flip_method": 2,
            "width": 640,
            "height": 480,
            "fps": 15,
            "frame_age_ms": 28,
            "error": None,
        },
        "sidecar_depth_imu": {
            "role": "sidecar_depth_imu",
            "type": "realsense",
            "label": "RealSense Sidecar Depth/IMU",
            "configured": True,
            "enabled": True,
            "present": True,
            "healthy": True,
            "status": "available",
            "frame_available": False,
            "depth_available": True,
            "imu_available": True,
            "source": "realsense",
            "used_for": ["obstacle_stop", "state_context"],
            "depth_status": "available",
            "imu_status": "available",
            "depth_frame_age_ms": 41,
            "imu_frame_age_ms": 19,
            "imu_data": {
                "accel": [0.01, -0.02, 0.98],
                "gyro": [0.001, 0.002, 0.003],
            },
            "width": 640,
            "height": 480,
            "fps": 15,
            "frame_age_ms": 32,
            "error": None,
        },
        "depth": {
            "role": "depth",
            "type": "realsense",
            "label": "Depth Sidecar",
            "configured": True,
            "enabled": True,
            "present": True,
            "healthy": True,
            "status": "available",
            "frame_available": True,
            "depth_available": True,
            "imu_available": False,
            "source": "realsense",
            "frame_age_ms": 41,
            "used_for": ["obstacle_stop"],
            "depth_status": "available",
            "imu_status": "disabled",
            "error": None,
        },
        "imu": {
            "role": "imu",
            "type": "realsense",
            "label": "IMU Sidecar",
            "configured": True,
            "enabled": True,
            "present": True,
            "healthy": True,
            "status": "available",
            "frame_available": True,
            "depth_available": False,
            "imu_available": True,
            "source": "realsense",
            "frame_age_ms": 19,
            "used_for": ["state_context"],
            "depth_status": "disabled",
            "imu_status": "available",
            "error": None,
        },
        "rear_preview": {
            "role": "rear_preview",
            "type": "csi",
            "label": "CAM1 Rear Preview",
            "configured": True,
            "enabled": False,
            "present": False,
            "healthy": False,
            "status": "disabled",
            "frame_available": False,
            "depth_available": False,
            "imu_available": False,
            "source": "cam1",
            "used_for": ["rear_preview_only"],
            "sensor_id": 1,
            "flip_method": 2,
            "width": 640,
            "height": 480,
            "fps": 15,
            "frame_age_ms": None,
            "error": None,
        },
        "rear": {
            "role": "rear",
            "type": "csi",
            "label": "CAM1 Rear Preview",
            "configured": True,
            "enabled": False,
            "healthy": False,
            "present": False,
            "status": "disabled",
            "frame_available": False,
            "depth_available": False,
            "imu_available": False,
            "source": "cam1",
            "used_for": ["rear_preview_only"],
            "frame_age_ms": None,
            "error": None,
        },
    }

@app.get("/api/cars")
def get_cars():
    """List all configured cars with their last known status."""
    return [_safe_car(c) for c in db.cars.values()]

@app.post("/api/test-client")
def add_test_client():
    """Adds a simulated client for UI testing."""
    client_id = f"test-client-{uuid.uuid4().hex[:8]}"
    mission_snapshot = {
        "enabled": True,
        "route_name": "expo_route",
        "state": "RUNNING",
        "stop_reason": None,
        "tag_detector_status": "available",
        "control_model_status": "available",
        "depth_status": "available",
        "obstacle_distance_m": 1.25,
        "start_tag_seen": True,
        "checkpoint_seen": True,
        "goal_seen": False,
        "expected_next_tag": 30,
        "last_tag_id": 20,
        "last_tag_seen_ts": time.time() - 5.0,
        "last_checkpoint_seen_ts": time.time() - 5.0,
        "last_goal_seen_ts": None,
        "goal_approach_since": None,
        "state_changed_at": time.time() - 15.0,
        "tag_cooldown_s": 1.25,
        "tag_detect_every_n_frames": 3,
        "depth_stop": {
            "enabled": True,
            "threshold_m": 0.6,
            "roi": {"x": 0.35, "y": 0.35, "w": 0.30, "h": 0.30},
        },
        "tag_ids": {"start_home": 10, "checkpoint": 20, "goal": 30},
    }
    db.cars[client_id] = {
        "id": client_id,
        "name": f"Test Racer {random.randint(1, 99)}",
        "ip": "127.0.0.1",
        "port": 0,
        "password": "changeme",
        "status": "online",
        "details": {
            "running": True,
            "paused": False,
            "config": {
                "device": "cuda",
                "architecture": "resnet101",
                "cameras": _test_client_role_cameras(),
                "control_model_type": "tensorrt",
                "control_model": "checkpoints/best_model_trt.pth",
                "detection_model": "yolov8n.pt",
                "throttle_mode": "fixed",
                "fixed_throttle_value": 0.22,
                "action_loop": ["control", "detection", "api"],
                "preprocess_profile": "cam0_fisheye_v1",
                "mission": mission_snapshot,
            },
            "state": {
                "fps": 30,
                "location": {
                    "x": 0,
                    "y": 0,
                    "theta": 0,
                    "imu": {
                        "accel": [0.01, -0.02, 0.98],
                        "gyro": [0.001, 0.002, 0.003],
                    },
                },
                "mission": mission_snapshot,
                "sensors": _test_client_sensor_state(),
                "specs": {
                    "device": "NVIDIA Jetson Nano",
                    "cpu_ram": "4-core ARMv57 / 4GB LPDDR4",
                    "cameras": [
                        {"role": "primary_rgb", "type": "csi", "width": 640, "height": 480, "fps": 15, "sensor_id": 0, "flip_method": 2},
                        {"role": "sidecar_depth_imu", "type": "realsense", "width": 640, "height": 480, "fps": 15, "enabled": True},
                        {"role": "rear_preview", "type": "csi", "width": 640, "height": 480, "fps": 15, "sensor_id": 1, "flip_method": 2, "enabled": False},
                    ],
                    "inference": "CUDA (GPU)",
                    "resnet_version": "ResNet-101",
                    "yolo_version": "YOLOv8n"
                }
            }
        },
        "geo": {"city": "Sim City", "country": "Virtual Land"},
        "type": "virtual",
        "fernet": None,
    }
    # Seed some fake log entries for the test client
    now = time.time()
    _append_log(client_id, [
        {"timestamp": now - 30, "level": "INFO",  "message": "Car system starting up…"},
        {"timestamp": now - 25, "level": "INFO",  "message": "Camera initialised (640×480 @ 30 fps)."},
        {"timestamp": now - 20, "level": "INFO",  "message": "Model loaded: ResNet-101 (TensorRT)."},
        {"timestamp": now - 15, "level": "INFO",  "message": "YOLO detector ready: yolov8n."},
        {"timestamp": now - 10, "level": "DEBUG", "message": "IMU connected on /dev/ttyUSB0."},
        {"timestamp": now -  5, "level": "INFO",  "message": "Action loop started: control, detection, api."},
        {"timestamp": now -  2, "level": "WARN",  "message": "Low light – confidence may drop."},
        {"timestamp": now,      "level": "INFO",  "message": "Running at 30 fps."},
    ])
    return {"status": "added", "id": client_id}

@app.post("/api/cars")
def add_car(car: NewClient):
    # Sanitize inputs
    ip = car.ip.strip()
    if ip.startswith("http://"): ip = ip.replace("http://", "")
    if ip.startswith("https://"): ip = ip.replace("https://", "")
    if "/" in ip: ip = ip.split("/")[0]

    key = f"{ip}:{car.port}"
    if key in db.cars:
        return {"status": "exists", "car": _safe_car(db.cars[key])}

    db.cars[key] = {
        "id": key,
        "name": car.name.strip(),
        "ip": ip,
        "port": car.port,
        "password": car.password,
        "status": "unknown",
        "details": {},
        "geo": get_geo_info(ip),
        "fernet": _make_fernet(car.password),
    }
    return {"status": "added", "car": _safe_car(db.cars[key])}

@app.delete("/api/cars/{car_id}")
def remove_car(car_id: str):
    if car_id in db.cars:
        del db.cars[car_id]
        return {"status": "removed"}
    raise HTTPException(status_code=404, detail="Car not found")

@app.get("/api/cars/{car_id}/status")
def proxy_status(car_id: str):
    """Fetch live status from the car and update DB."""
    if car_id not in db.cars:
        raise HTTPException(status_code=404, detail="Car not found")

    car = db.cars[car_id]
    
    if car.get("type") in ("virtual", "websocket"):
        return _redact_sensitive(car.get("details", {}))

    url = f"http://{car['ip']}:{car['port']}/status"
    headers = {"X-Api-Key": car.get("password", "changeme")}

    try:
        r = requests.get(url, headers=headers, timeout=3)
        if r.status_code == 200:
            data = r.json()
            is_running = data.get("running", False)
            car["status"] = "online" if is_running else "stopped"
            car["details"] = data
            return _redact_sensitive(data)
    except Exception as e:
        err_msg = str(e)
        print(f"Error reaching {url}: {err_msg}")
        car["status"] = "offline"
        car["details"] = {"error": err_msg}
        return {"running": False, "state": {}, "error": err_msg}

    return {"running": False, "state": {}, "error": "unreachable"}

@app.post("/api/cars/{car_id}/start")
async def proxy_start(car_id: str):
    return await send_car_command(car_id, "start")

@app.post("/api/cars/{car_id}/stop")
async def proxy_stop(car_id: str):
    return await send_car_command(car_id, "stop")

@app.get("/api/cars/{car_id}/logs")
def get_car_logs(car_id: str, since: float = 0.0):
    """Return buffered log entries for a car, optionally filtered by timestamp."""
    if car_id not in db.cars:
        raise HTTPException(status_code=404, detail="Car not found")
    buf = db.logs.get(car_id, deque())
    entries = [e for e in buf if e["timestamp"] > since]
    return {"car_id": car_id, "logs": entries}

@app.delete("/api/cars/{car_id}/logs")
def clear_car_logs(car_id: str):
    """Clear all buffered log entries for a car."""
    if car_id not in db.cars:
        raise HTTPException(status_code=404, detail="Car not found")
    db.logs[car_id] = deque(maxlen=_MAX_LOG_ENTRIES)
    return {"status": "cleared"}

@app.post("/api/cars/{car_id}/pause")
async def proxy_pause(car_id: str):
    return await send_car_command(car_id, "pause", {"duration": None})

@app.post("/api/cars/{car_id}/resume")
async def proxy_resume(car_id: str):
    return await send_car_command(car_id, "resume")

@app.post("/api/cars/{car_id}/settings")
async def proxy_settings(car_id: str, settings: ClientSettings):
    return await send_car_command(car_id, "update_settings", settings.dict())

@app.post("/api/cars/{car_id}/config")
async def proxy_config(car_id: str, config_body: DeployConfig):
    # For WS, we just wrap the config in a "configure" command
    return await send_car_command(car_id, "configure", config_body.config)

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@app.get("/api/host-info")
def get_host_info():
    return {"ip": get_local_ip(), "port": 3000}

# --- Statics & Root ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))

def main():
    host_ip = get_local_ip()
    print("="*40)
    print(" JETRACER HOST SERVER LIVE")
    print(f" Web Interface: http://localhost:3000")
    print(f" Use this IP for Clients: {host_ip}")
    print("="*40)
    
    uvicorn.run(app, host="0.0.0.0", port=3000)

if __name__ == "__main__":
    main()

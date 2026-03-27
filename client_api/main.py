from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import sys
import os
import asyncio
import websockets
import json
import threading
import hmac
import hashlib
import base64
import os as _os
import time

# ── Encryption ──────────────────────────────────────────────────────────────
try:
    from cryptography.fernet import Fernet, InvalidToken
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False
    logging.warning("[Security] 'cryptography' package not installed – messages will NOT be encrypted!")

def _derive_fernet_key(password: str) -> bytes:
    """Derive a 32-byte Fernet key from a plain-text password via SHA-256."""
    digest = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(digest)

def _make_fernet(password: str):
    if not _FERNET_AVAILABLE:
        return None
    return Fernet(_derive_fernet_key(password))

def encrypt_payload(data: dict, fernet) -> str:
    """Encrypt a dict to a base64 Fernet token string."""
    if fernet is None:
        return json.dumps(data)
    raw = json.dumps(data).encode()
    return fernet.encrypt(raw).decode()

def decrypt_payload(token: str, fernet) -> dict:
    """Decrypt a Fernet token string to a dict."""
    if fernet is None:
        return json.loads(token)
    raw = fernet.decrypt(token.encode())
    return json.loads(raw)

def _hmac_challenge(password: str, challenge: str) -> str:
    """Produce HMAC-SHA256 hex digest of challenge using password as key."""
    return hmac.new(password.encode(), challenge.encode(), hashlib.sha256).hexdigest()

# Fix for "ImportError: attempted relative import..." when running directly
try:
    from .car import car
except ImportError:
    # If running as script, add current dir to sys.path if not present
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from car import car

app = FastAPI(title="JetRacer Client API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClientAPI")

# ── In-memory log buffer sent to host with each WS state update ─────────────
import threading as _threading
from collections import deque as _deque

_LOG_BUFFER_MAX = 200  # max entries held before oldest are dropped
_log_buffer: _deque = _deque(maxlen=_LOG_BUFFER_MAX)
_log_buffer_lock = _threading.Lock()

class _WsLogHandler(logging.Handler):
    """Captures log records into _log_buffer so they can be forwarded to the host."""
    def emit(self, record: logging.LogRecord):
        entry = {
            "timestamp": record.created,
            "level":     record.levelname,
            "message":   self.format(record),
        }
        with _log_buffer_lock:
            _log_buffer.append(entry)

_ws_log_handler = _WsLogHandler()
_ws_log_handler.setFormatter(logging.Formatter("%(name)s – %(message)s"))
logging.getLogger().addHandler(_ws_log_handler)  # attach to root so all loggers feed it


def _drain_log_buffer() -> list:
    """Return and clear all pending log entries (thread-safe)."""
    with _log_buffer_lock:
        entries = list(_log_buffer)
        _log_buffer.clear()
    return entries

# ── Runtime password / fernet (updated when /configure is called) ────────────
_current_password: str = "changeme"
_current_fernet = _make_fernet(_current_password)

def _update_security(password: str):
    global _current_password, _current_fernet
    _current_password = password
    _current_fernet = _make_fernet(password)

def _verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Dependency: reject requests whose X-Api-Key header doesn't match."""
    if not hmac.compare_digest(x_api_key or "", _current_password):
        raise HTTPException(status_code=401, detail="Unauthorized")

class CameraConfig(BaseModel):
    type: str # 'opencv' or 'realsense'
    index: Optional[int] = 0
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[int] = 30

class ClientConfig(BaseModel):
    device: str = 'cuda'
    architecture: str = 'resnet18'
    cameras: List[CameraConfig] = []
    control_model_type: str = 'pytorch'
    control_model: str
    detection_model: str
    throttle_mode: str = 'fixed' # 'fixed' or 'ai'
    fixed_throttle_value: float = 0.22
    action_loop: List[str] = ['control', 'detection', 'api']
    ip: str = "0.0.0.0"
    port: int = 8000
    password: str = "changeme"

class PauseRequest(BaseModel):
    duration: Optional[float] = None

class StateUpdate(BaseModel):
    location: Optional[Any] = None

class UpdateSettingsRequest(BaseModel):
    throttle_mode: Optional[str] = None
    fixed_throttle_value: Optional[float] = None

class NavigateRequest(BaseModel):
    x: float
    y: float

@app.post("/configure")
def configure_car(config: ClientConfig, _: None = Depends(_verify_api_key)):
    try:
        # Update runtime password/fernet if a new password was sent
        _update_security(config.password)
        car.configure(config.dict())
        return {"status": "configured"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
def start_car(_: None = Depends(_verify_api_key)):
    car.start_logic()
    return {"status": "started"}

@app.post("/stop")
def stop_car(_: None = Depends(_verify_api_key)):
    car.stop_logic()
    return {"status": "stopped"}

@app.post("/pause")
def pause_car(req: PauseRequest, _: None = Depends(_verify_api_key)):
    car.pause(req.duration)
    return {"status": "paused", "duration": req.duration}

@app.post("/resume")
def resume_car(_: None = Depends(_verify_api_key)):
    car.resume()
    return {"status": "resumed"}

@app.post("/navigate")
def set_destination(req: NavigateRequest, _: None = Depends(_verify_api_key)):
    """Override control to navigate to a specific generic coordinate [x, y]"""
    car.target_dest = (req.x, req.y)
    return {"status": "navigating", "target": car.target_dest}

@app.post("/navigate/cancel")
def cancel_navigation(_: None = Depends(_verify_api_key)):
    car.target_dest = None
    return {"status": "cancelled"}

@app.get("/status")
def get_status(_: None = Depends(_verify_api_key)):
    return {
        "running": car.running,
        "paused": car.paused,
        "state": car.state
    }

@app.post("/update_state")
def update_state(update: StateUpdate, _: None = Depends(_verify_api_key)):
    if update.location:
        car.state["location"] = update.location
    return {"status": "updated", "state": car.state}

@app.post("/update_settings")
def update_settings(req: UpdateSettingsRequest, _: None = Depends(_verify_api_key)):
    if req.throttle_mode:
        car.set_throttle_mode(req.throttle_mode, req.fixed_throttle_value)
    elif req.fixed_throttle_value:
        car.fixed_throttle = req.fixed_throttle_value
    return {"status": "updated", "throttle_mode": car.config.get("throttle_mode"), "fixed_throttle": car.fixed_throttle}

@app.get("/")
def health_check():
    return {"status": "ok", "service": "JetRacer Client", "version": "1.0"}

# ── Live Video Stream WebSocket ──────────────────────────────────────────────
@app.websocket("/ws/video")
async def video_stream_endpoint(websocket: WebSocket):
    """
    Streams JPEG frames to the host server (or any authenticated client).
    Protocol:
      1. Client (us) accepts the connection.
      2. Remote sends JSON {"auth": "<password>"} as first message.
      3. We reply {"status": "ok"} or close with 4401.
      4. We then push binary JPEG frames in a loop.
    """
    await websocket.accept()
    try:
        # --- Authentication handshake ---
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            auth_data = json.loads(auth_msg)
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await websocket.close(code=4401)
            return

        provided = auth_data.get("auth", "")
        if not hmac.compare_digest(provided, _current_password):
            await websocket.send_text(json.dumps({"status": "unauthorized"}))
            await websocket.close(code=4401)
            return

        await websocket.send_text(json.dumps({"status": "ok"}))

        # --- Stream frames ---
        import cv2 as _cv2
        cap = None
        try:
            req_index = auth_data.get("camera_index")
            if req_index is not None:
                cam_index = int(req_index)
            else:
                cam_cfg = (car.config.get("cameras") or [{}])
                cam_index = cam_cfg[0].get("index", 0) if cam_cfg else 0
            cap = _cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "camera_unavailable"}))
                await websocket.close(code=1011)
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.05)
                    continue

                # Encode frame as JPEG
                _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = buf.tobytes()

                # Encrypt the bytes if fernet is available
                if _current_fernet is not None:
                    payload = _current_fernet.encrypt(jpeg_bytes)
                else:
                    payload = jpeg_bytes

                await websocket.send_bytes(payload)
                await asyncio.sleep(1 / 15)  # ~15 FPS stream

        except WebSocketDisconnect:
            pass
        finally:
            if cap and cap.isOpened():
                cap.release()
    except WebSocketDisconnect:
        pass

# --- WebSocket Client Logic ---
async def websocket_loop(host_ip: str, client_name: str = "Jetson"):
    """
    Connects to the host server's /ws/car/<name> endpoint.

    Authentication + Encryption protocol
    ─────────────────────────────────────
    1. Server sends: {"type": "challenge", "nonce": "<random_hex>"}
    2. Client replies: {"type": "auth", "response": HMAC(password, nonce)}
    3. Server sends: {"type": "auth_ok"} or closes with 4401.
    4. All subsequent messages are Fernet-encrypted JSON.
    """
    uri = f"ws://{host_ip}:3000/ws/car/{client_name}"
    print(f"[WS] Connecting to {uri}...")

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("[WS] Connected – performing auth handshake…")

                # ── Step 1: receive challenge ────────────────────────────
                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                challenge_msg = json.loads(raw)
                if challenge_msg.get("type") != "challenge":
                    print("[WS] Unexpected first message – closing.")
                    break
                nonce = challenge_msg["nonce"]

                # ── Step 2: send HMAC response ───────────────────────────
                response = _hmac_challenge(_current_password, nonce)
                await websocket.send(json.dumps({"type": "auth", "response": response}))

                # ── Step 3: wait for auth_ok ─────────────────────────────
                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                ack = json.loads(raw)
                if ack.get("type") != "auth_ok":
                    print(f"[WS] Auth rejected: {ack}. Retrying in 10s…")
                    await asyncio.sleep(10)
                    continue

                print("[WS] Authenticated ✓  (encrypted channel active)")

                # ── Step 4: start encrypted state/command loop ───────────
                async def send_state():
                    try:
                        while True:
                            state = {
                                "running": car.running,
                                "paused": car.paused,
                                "state": car.state,
                                "config": car.config,
                                "logs": _drain_log_buffer(),  # forward buffered log records
                            }
                            payload = encrypt_payload(state, _current_fernet)
                            await websocket.send(payload)
                            await asyncio.sleep(0.5)  # 2 Hz updates
                    except websockets.ConnectionClosed:
                        print("[WS] Sender task stopping: connection closed.")
                    except Exception as e:
                        print(f"[WS] Sender task error: {e}")

                sender_task = asyncio.ensure_future(send_state())

                try:
                    async for message in websocket:
                        try:
                            cmd = decrypt_payload(message, _current_fernet)
                        except Exception:
                            logger.warning("[WS] Failed to decrypt/parse message – skipping.")
                            continue

                        logger.info(f"[WS] Received: {cmd}")
                        action = cmd.get("command")

                        if action == "start":
                            car.start_logic()
                        elif action == "stop":
                            car.stop_logic()
                        elif action == "pause":
                            car.pause(cmd.get("duration"))
                        elif action == "resume":
                            car.resume()
                        elif action == "update_settings":
                            if "throttle_mode" in cmd:
                                car.set_throttle_mode(cmd["throttle_mode"])
                            if "fixed_throttle_value" in cmd:
                                car.fixed_throttle = cmd["fixed_throttle_value"]
                        elif action == "configure":
                            threading.Thread(target=car.configure, args=(cmd,)).start()

                except websockets.ConnectionClosed:
                    print("[WS] Connection lost.")
                    sender_task.cancel()
                    break
        except Exception as e:
            print(f"[WS] Connection failed: {e}. Retrying in 5s…")
            await asyncio.sleep(5)

def main():
    import socket
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Host IP for WebSocket connection")
    parser.add_argument("--name", type=str, default="Jetson", help="Car Name")
    args = parser.parse_args()

    # Get local IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    
    print("="*40)
    print(f" Jetracer Client API Live")
    print(f" Listening on: http://0.0.0.0:8000")
    print(f" LAN IP Hints: {IP}")
    if args.host:
        print(f" Connecting to Host: {args.host}")
    print("="*40)
    
    # Run API and WebSocket in parallel
    loop = asyncio.new_event_loop()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", loop=loop)
    server = uvicorn.Server(config)
    
    # We need to run the server in the asyncio loop along with our WS client
    async def run_system():
        tasks = [server.serve()]
        if args.host:
            tasks.append(websocket_loop(args.host, args.name))
        await asyncio.gather(*tasks)

    loop.run_until_complete(run_system())

if __name__ == "__main__":
    main()

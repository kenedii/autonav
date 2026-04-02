import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import threading

import uvicorn
import websockets
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


try:
    from cryptography.fernet import Fernet
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False
    logging.warning("[Security] 'cryptography' package not installed - messages will NOT be encrypted!")


def _derive_fernet_key(password):
    digest = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def _make_fernet(password):
    if not _FERNET_AVAILABLE:
        return None
    return Fernet(_derive_fernet_key(password))


def encrypt_payload(data, fernet):
    if fernet is None:
        return json.dumps(data)
    raw = json.dumps(data).encode()
    return fernet.encrypt(raw).decode()


def decrypt_payload(token, fernet):
    if fernet is None:
        return json.loads(token)
    raw = fernet.decrypt(token.encode())
    return json.loads(raw)


def _hmac_challenge(password, challenge):
    return hmac.new(password.encode(), challenge.encode(), hashlib.sha256).hexdigest()


try:
    from .car import car
except ImportError:
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

import threading as _threading
from collections import deque as _deque

_LOG_BUFFER_MAX = 200
_log_buffer = _deque(maxlen=_LOG_BUFFER_MAX)
_log_buffer_lock = _threading.Lock()


class _WsLogHandler(logging.Handler):
    def emit(self, record):
        entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "message": self.format(record),
        }
        with _log_buffer_lock:
            _log_buffer.append(entry)


_ws_log_handler = _WsLogHandler()
_ws_log_handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
logging.getLogger().addHandler(_ws_log_handler)


def _drain_log_buffer():
    with _log_buffer_lock:
        entries = list(_log_buffer)
        _log_buffer.clear()
    return entries


_current_password = "changeme"
_current_fernet = _make_fernet(_current_password)


def _update_security(password):
    global _current_password, _current_fernet
    _current_password = password
    _current_fernet = _make_fernet(password)


def _verify_api_key(x_api_key=Header(default=None)):
    if not hmac.compare_digest(x_api_key or "", _current_password):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _sanitize_json_value(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key == "password":
                continue
            sanitized[key] = _sanitize_json_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _sanitize_json_value(value.tolist())
        except Exception:
            pass
    return value


def _sanitized_config(config):
    return _sanitize_json_value(config or {})


def _build_state_payload():
    return {
        "running": bool(car.running),
        "paused": bool(car.paused),
        "state": _sanitize_json_value(car.state),
        "config": _sanitized_config(car.config),
        "logs": _sanitize_json_value(_drain_log_buffer()),
    }


class CameraConfig(BaseModel):
    type: str
    role: Optional[str] = None
    enabled: Optional[bool] = True
    index: Optional[int] = 0
    sensor_id: Optional[int] = 0
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[int] = 30
    flip_method: Optional[int] = 0


class ClientConfig(BaseModel):
    device: str = "cuda"
    architecture: str = "resnet18"
    cameras: List[CameraConfig] = []
    control_model_type: str = "pytorch"
    control_model: Optional[str] = None
    detection_model: Optional[str] = None
    throttle_mode: str = "fixed"
    fixed_throttle_value: float = 0.22
    preprocess_profile: Optional[str] = None
    action_loop: List[str] = ["control", "detection", "api"]
    mission: Optional[Dict[str, Any]] = None
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
def configure_car(config: ClientConfig, _=Depends(_verify_api_key)):
    try:
        config_dict = config.dict()
        _update_security(config_dict.get("password", _current_password))
        car.configure(config_dict)
        return {"status": "configured"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/start")
def start_car(_=Depends(_verify_api_key)):
    car.start_logic()
    return {"status": "started"}


@app.post("/stop")
def stop_car(_=Depends(_verify_api_key)):
    car.stop_logic()
    return {"status": "stopped"}


@app.post("/pause")
def pause_car(req: PauseRequest, _=Depends(_verify_api_key)):
    car.pause(req.duration)
    return {"status": "paused", "duration": req.duration}


@app.post("/resume")
def resume_car(_=Depends(_verify_api_key)):
    car.resume()
    return {"status": "resumed"}


@app.post("/navigate")
def set_destination(req: NavigateRequest, _=Depends(_verify_api_key)):
    car.target_dest = (req.x, req.y)
    return {"status": "navigating", "target": car.target_dest}


@app.post("/navigate/cancel")
def cancel_navigation(_=Depends(_verify_api_key)):
    car.target_dest = None
    return {"status": "cancelled"}


@app.get("/status")
def get_status(_=Depends(_verify_api_key)):
    return {
        "running": bool(car.running),
        "paused": bool(car.paused),
        "state": _sanitize_json_value(car.state),
    }


@app.post("/update_state")
def update_state(update: StateUpdate, _=Depends(_verify_api_key)):
    if update.location:
        car.state["location"] = _sanitize_json_value(update.location)
    return {"status": "updated", "state": _sanitize_json_value(car.state)}


@app.post("/update_settings")
def update_settings(req: UpdateSettingsRequest, _=Depends(_verify_api_key)):
    if req.throttle_mode is not None:
        car.set_throttle_mode(req.throttle_mode, req.fixed_throttle_value)
    elif req.fixed_throttle_value is not None:
        car.fixed_throttle = float(req.fixed_throttle_value)
    return {
        "status": "updated",
        "throttle_mode": car.config.get("throttle_mode"),
        "fixed_throttle": car.fixed_throttle,
    }


@app.get("/")
def health_check():
    return {"status": "ok", "service": "JetRacer Client", "version": "1.0"}


@app.websocket("/ws/video")
async def video_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            auth_data = json.loads(auth_msg)
        except (asyncio.TimeoutError, ValueError):
            await websocket.close(code=4401)
            return

        provided = auth_data.get("auth", "")
        if not hmac.compare_digest(provided, _current_password):
            await websocket.send_text(json.dumps({"status": "unauthorized"}))
            await websocket.close(code=4401)
            return

        await websocket.send_text(json.dumps({"status": "ok"}))

        while True:
            jpeg_bytes = car.get_latest_preview_jpeg()
            if not jpeg_bytes:
                await asyncio.sleep(0.05)
                continue

            if _current_fernet is not None:
                payload = _current_fernet.encrypt(jpeg_bytes)
            else:
                payload = jpeg_bytes
            await websocket.send_bytes(payload)
            await asyncio.sleep(1.0 / 15.0)
    except WebSocketDisconnect:
        pass


async def websocket_loop(host_ip, client_name="Jetson"):
    uri = "ws://%s:3000/ws/car/%s" % (host_ip, client_name)
    print("[WS] Connecting to %s..." % uri)

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("[WS] Connected - performing auth handshake...")

                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                challenge_msg = json.loads(raw)
                if challenge_msg.get("type") != "challenge":
                    print("[WS] Unexpected first message - closing.")
                    break
                nonce = challenge_msg["nonce"]

                response = _hmac_challenge(_current_password, nonce)
                await websocket.send(json.dumps({"type": "auth", "response": response}))

                raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                ack = json.loads(raw)
                if ack.get("type") != "auth_ok":
                    print("[WS] Auth rejected: %s. Retrying in 10s..." % ack)
                    await asyncio.sleep(10)
                    continue

                print("[WS] Authenticated")

                async def send_state():
                    try:
                        while True:
                            payload = encrypt_payload(_build_state_payload(), _current_fernet)
                            await websocket.send(payload)
                            await asyncio.sleep(0.5)
                    except websockets.ConnectionClosed:
                        print("[WS] Sender task stopping: connection closed.")
                    except Exception as exc:
                        print("[WS] Sender task error: %s" % exc)

                sender_task = asyncio.ensure_future(send_state())

                try:
                    async for message in websocket:
                        try:
                            cmd = decrypt_payload(message, _current_fernet)
                        except Exception:
                            logger.warning("[WS] Failed to decrypt/parse message - skipping.")
                            continue

                        logger.info("[WS] Received: %s", cmd)
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
                                car.set_throttle_mode(cmd["throttle_mode"], cmd.get("fixed_throttle_value"))
                            elif "fixed_throttle_value" in cmd:
                                car.fixed_throttle = float(cmd["fixed_throttle_value"])
                        elif action == "configure":
                            payload = dict(cmd)
                            payload.pop("command", None)
                            if payload.get("password"):
                                _update_security(payload["password"])
                            threading.Thread(
                                target=car.configure,
                                args=(payload,),
                                daemon=True,
                            ).start()

                except websockets.ConnectionClosed:
                    print("[WS] Connection lost.")
                    sender_task.cancel()
                    break
        except Exception as exc:
            print("[WS] Connection failed: %s. Retrying in 5s..." % exc)
            await asyncio.sleep(5)


def main():
    import argparse
    import socket

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Host IP for WebSocket connection")
    parser.add_argument("--name", type=str, default="Jetson", help="Car Name")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        ip_address = sock.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        sock.close()

    print("=" * 40)
    print(" Jetracer Client API Live")
    print(" Listening on: http://0.0.0.0:8000")
    print(" LAN IP Hints: %s" % ip_address)
    if args.host:
        print(" Connecting to Host: %s" % args.host)
    print("=" * 40)

    loop = asyncio.new_event_loop()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", loop=loop)
    server = uvicorn.Server(config)

    async def run_system():
        tasks = [server.serve()]
        if args.host:
            tasks.append(websocket_loop(args.host, args.name))
        await asyncio.gather(*tasks)

    loop.run_until_complete(run_system())


if __name__ == "__main__":
    main()

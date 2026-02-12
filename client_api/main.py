from fastapi import FastAPI, HTTPException, Body
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
def configure_car(config: ClientConfig):
    try:
        car.configure(config.dict())
        return {"status": "configured"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
def start_car():
    car.start_logic()
    return {"status": "started"}

@app.post("/stop")
def stop_car():
    car.stop_logic()
    return {"status": "stopped"}

@app.post("/pause")
def pause_car(req: PauseRequest):
    car.pause(req.duration)
    return {"status": "paused", "duration": req.duration}

@app.post("/resume")
def resume_car():
    car.resume()
    return {"status": "resumed"}

@app.post("/navigate")
def set_destination(req: NavigateRequest):
    """Override control to navigate to a specific generic coordinate [x, y]"""
    car.target_dest = (req.x, req.y)
    return {"status": "navigating", "target": car.target_dest}

@app.post("/navigate/cancel")
def cancel_navigation():
    car.target_dest = None
    return {"status": "cancelled"}

@app.get("/status")
def get_status():
    return {
        "running": car.running,
        "paused": car.paused,
        "state": car.state
    }

@app.post("/update_state")
def update_state(update: StateUpdate):
    if update.location:
        car.state["location"] = update.location
    return {"status": "updated", "state": car.state}

@app.post("/update_settings")
def update_settings(req: UpdateSettingsRequest):
    if req.throttle_mode:
        car.set_throttle_mode(req.throttle_mode, req.fixed_throttle_value)
    elif req.fixed_throttle_value:
        car.fixed_throttle = req.fixed_throttle_value
    return {"status": "updated", "throttle_mode": car.config.get("throttle_mode"), "fixed_throttle": car.fixed_throttle}

@app.get("/")
def health_check():
    return {"status": "ok", "service": "JetRacer Client", "version": "1.0"}

# --- WebSocket Client Logic ---
async def websocket_loop(host_ip: str, client_name: str="Jetson"):
    uri = f"ws://{host_ip}:3000/ws/car/{client_name}"
    print(f"[WS] Connecting to {uri}...")
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("[WS] Connected!")
                
                # Separate Task for Sending State
                async def send_state():
                    try:
                        while True:
                            state = {
                                "running": car.running,
                                "paused": car.paused,
                                "state": car.state,
                                "config": car.config
                            }
                            await websocket.send(json.dumps(state))
                            await asyncio.sleep(0.5) # 2Hz updates
                    except websockets.ConnectionClosed:
                        print("[WS] Sender task stopping: Connection closed.")
                    except Exception as e:
                        print(f"[WS] Sender task error: {e}")

                sender_task = asyncio.ensure_future(send_state())
                
                # Main Loop: Receive Commands
                try:
                    async for message in websocket:
                        cmd = json.loads(message)
                        logger.info(f"[WS] Received: {cmd}")
                        
                        action = cmd.get("command")
                        
                        if action == "start": car.start_logic()
                        elif action == "stop": car.stop_logic()
                        elif action == "pause": car.pause(cmd.get("duration"))
                        elif action == "resume": car.resume()
                        elif action == "update_settings":
                            if "throttle_mode" in cmd: car.set_throttle_mode(cmd["throttle_mode"])
                            if "fixed_throttle_value" in cmd: car.fixed_throttle = cmd["fixed_throttle_value"]
                        elif action == "configure":
                             # Run in thread to not block WS loop
                             threading.Thread(target=car.configure, args=(cmd,)).start()
                             
                except websockets.ConnectionClosed:
                    print("[WS] Connection lost.")
                    sender_task.cancel()
                    break
        except Exception as e:
            print(f"[WS] Connection failed: {e}. Retrying in 5s...")
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

from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import uvicorn
import json
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware

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

class ClientSettings(BaseModel):
    throttle_mode: Optional[str] = None
    fixed_throttle_value: Optional[float] = None

class DeployConfig(BaseModel):
    config: Dict[str, Any]

# --- In-Memory Database ---
# (In a real app, use a database. Here we use a global list)
class CarDb:
    def __init__(self):
        self.cars: Dict[str, Dict] = {} 
        # Structure: { 
        #   "ip:port": { "name": "...", "ip": "...", "port": 8000, "status": "offline", "last_seen": 0 }
        # }

db = CarDb()

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
        # Map car_id (ip:port usually, or name) -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, car_id: str):
        await websocket.accept()
        self.active_connections[car_id] = websocket

    def disconnect(self, car_id: str):
        if car_id in self.active_connections:
            del self.active_connections[car_id]

    async def send_command(self, car_id: str, command: dict):
        if car_id in self.active_connections:
            await self.active_connections[car_id].send_json(command)
            return True
        return False

manager = ConnectionManager()

@app.websocket("/ws/car/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
    # We use the client's socket info to form an ID, or just use the name if unique
    # For now, let's trust the name or append IP
    ip = websocket.client.host
    car_id = f"{ip}:WS" # Distinguish from HTTP registered cars
    
    # Auto-register if not in DB
    if car_id not in db.cars:
        db.cars[car_id] = {
            "id": car_id,
            "name": client_name,
            "ip": ip,
            "port": 0, # WebSocket doesn't need port
            "status": "online",
            "details": {},
            "geo": get_geo_info(ip),
            "type": "websocket"
        }
    
    await manager.connect(websocket, car_id)
    try:
        while True:
            # Receive state updates from Client
            data = await websocket.receive_json()
            # Update DB
            if car_id in db.cars:
                db.cars[car_id]["details"] = data
                db.cars[car_id]["status"] = "online" if data.get("running") else "stopped"
    except WebSocketDisconnect:
        manager.disconnect(car_id)
        if car_id in db.cars:
            db.cars[car_id]["status"] = "offline"

# --- Hybrid Proxy Helpers ---
async def send_car_command(car_id: str, command: str, payload: dict = {}):
    if car_id not in db.cars: raise HTTPException(404, "Car not found")
    car = db.cars[car_id]
    
    # WebSocket Client
    if car.get("type") == "websocket":
        cmd_payload = {"command": command}
        cmd_payload.update(payload)
        success = await manager.send_command(car_id, cmd_payload)
        if not success: raise HTTPException(503, "Car disconnected")
        return {"status": "sent_via_ws"}
    
    # Legacy HTTP Client
    try:
        url = f"http://{car['ip']}:{car['port']}/{command}"
        if payload:
            requests.post(url, json=payload, timeout=2)
        else:
            requests.post(url, timeout=2)
        return {"status": "sent_via_http"}
    except Exception as e:
        raise HTTPException(500, f"HTTP Error: {e}")

# --- Endpoints ---

@app.get("/api/cars")
def get_cars():
    """List all configured cars with their last known status."""
    # We could do a quick ping in background or just return last known
    # For UI responsiveness, let's just return what we have, 
    # and let the UI trigger a 'refresh' 
    return list(db.cars.values())

@app.post("/api/cars")
def add_car(car: NewClient):
    # Sanitize inputs
    ip = car.ip.strip()
    if ip.startswith("http://"): ip = ip.replace("http://", "")
    if ip.startswith("https://"): ip = ip.replace("https://", "")
    if "/" in ip: ip = ip.split("/")[0]
    
    key = f"{ip}:{car.port}"
    if key in db.cars:
        return {"status": "exists", "car": db.cars[key]}
    
    db.cars[key] = {
        "id": key,
        "name": car.name.strip(),
        "ip": ip,
        "port": car.port,
        "status": "unknown",
        "details": {},
        "geo": get_geo_info(ip)
    }
    return {"status": "added", "car": db.cars[key]}

@app.delete("/api/cars/{car_id}")
def remove_car(car_id: str):
    # car_id is "ip:port"
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
    url = f"http://{car['ip']}:{car['port']}/status"
    
    try:
        r = requests.get(url, timeout=3) # Increased timeout
        if r.status_code == 200:
            data = r.json()
            # If running is True -> Online/Running
            # If running is False -> Online/Stopped
            # If request fails -> Offline
            
            is_running = data.get("running", False)
            car["status"] = "online" if is_running else "stopped"
            car["details"] = data
            return data
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
app.mount("/static", StaticFiles(directory="jetracer/host_app/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("jetracer/host_app/static/index.html")

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

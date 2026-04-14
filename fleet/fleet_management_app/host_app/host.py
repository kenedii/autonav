import requests
import json
import time

class CarAgent:
    def __init__(self, name, ip, port=8000, password="changeme"):
        self.name = name
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"
        self.password = password
        self.config = {}

    def _headers(self):
        """Return auth headers for every request."""
        return {"X-Api-Key": self.password}

    def check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/status", headers=self._headers(), timeout=2)
            if r.status_code == 200:
                print(f"[{self.name}] Online. Status: {r.json()['running']}")
                return True
            elif r.status_code == 401:
                print(f"[{self.name}] Authentication failed – wrong password?")
                return False
        except Exception:
            print(f"[{self.name}] Offline or not reachable at {self.ip}")
            return False

    def deploy_config(self, config_dict):
        # Override IP/Pass
        config_dict['ip'] = self.ip
        config_dict['port'] = self.port
        config_dict['password'] = self.password

        try:
            r = requests.post(
                f"{self.base_url}/configure",
                json=config_dict,
                headers=self._headers(),
            )
            if r.status_code == 200:
                print(f"[{self.name}] Configuration deployed.")
                self.config = config_dict
            else:
                print(f"[{self.name}] Deploy failed: {r.text}")
        except Exception as e:
            print(f"[{self.name}] Error deploying: {e}")

    def start(self):
        try:
            requests.post(f"{self.base_url}/start", headers=self._headers())
            print(f"[{self.name}] Started.")
        except:
            pass

    def stop(self):
        try:
            requests.post(f"{self.base_url}/stop", headers=self._headers())
            print(f"[{self.name}] Stopped.")
        except:
            pass

    def pause(self, duration=None):
        try:
            requests.post(f"{self.base_url}/pause", json={"duration": duration}, headers=self._headers())
            print(f"[{self.name}] Paused.")
        except:
            pass

    def resume(self):
        try:
            requests.post(f"{self.base_url}/resume", headers=self._headers())
            print(f"[{self.name}] Resumed.")
        except:
            pass

    def update_settings(self, throttle_mode=None, fixed_throttle_value=None):
        payload = {}
        if throttle_mode: payload['throttle_mode'] = throttle_mode
        if fixed_throttle_value: payload['fixed_throttle_value'] = fixed_throttle_value

        try:
            r = requests.post(f"{self.base_url}/update_settings", json=payload, headers=self._headers())
            print(f"[{self.name}] Settings updated: {r.json()}")
        except:
            pass

    def get_status(self):
        try:
            return requests.get(f"{self.base_url}/status", headers=self._headers()).json()
        except:
            return None

# Example Usage Script
if __name__ == "__main__":
    
    # 1. Define Cars (Specs)
    # Different cars might use different backends or cameras
    cars = [
        CarAgent("ExoRacer-1", "192.168.1.100"),
        CarAgent("ExoRacer-2", "192.168.1.101"),
    ]

    # 2. Define Configurations
    
    # Config for a Jetson Nano with TensorRT + RealSense
    jetson_config = {
        "device": "cuda",
        "cameras": [{"type": "realsense", "width": 848, "height": 480, "fps": 30}],
        "control_model_type": "tensorrt",
        "control_model": "checkpoints/best_model_trt.pth",
        "detection_model": "yolov8n.pt",
        "throttle_mode": "fixed",
        "fixed_throttle_value": 0.22,
        "action_loop": ["control", "detection"]
    }

    # Config for OrangePi with Rockchip NPU
    rockchip_config = {
        "device": "cpu",
        "cameras": [{"type": "opencv", "index": 0}],
        "control_model_type": "rockchip",
        "control_model": "best_model.rknn",
        "detection_model": "yolov8n.onnx",
        "throttle_mode": "ai", # Example
        "action_loop": ["control", "detection"]
    }

    print("--- Host Manager ---")
    
    # Simple CLI
    current_car = cars[0]
    
    while True:
        cmd = input(f"({current_car.name}) > ").strip().split()
        if not cmd: continue
        
        op = cmd[0].lower()
        
        if op == "switch":
            # switch 1
            idx = int(cmd[1])
            if idx < len(cars):
                current_car = cars[idx]
                
        elif op == "connect":
            current_car.check_connection()
            
        elif op == "deploy":
            # deploy jetson
            config_name = cmd[1] if len(cmd) > 1 else "jetson"
            if config_name == "jetson":
                current_car.deploy_config(jetson_config)
            elif config_name == "rockchip":
                current_car.deploy_config(rockchip_config)
                
        elif op == "start":
            current_car.start()
            
        elif op == "stop":
            current_car.stop()
            
        elif op == "status":
            st = current_car.get_status()
            if st:
                print(json.dumps(st, indent=2))
                # Check for distances in detections
                if "detections" in st.get("state", {}):
                    dets = st["state"]["detections"]
                    for d in dets:
                        dist = d.get("distance")
                        if dist:
                            print(f"!!! Object {d['class']} at {dist/1000.0:.2f} meters !!!")
                            
        elif op == "pause":
            dur = float(cmd[1]) if len(cmd) > 1 else None
            current_car.pause(dur)
            
        elif op == "resume":
            current_car.resume()

        elif op == "throttle":
            # throttle 0.25
            val = float(cmd[1])
            current_car.update_settings(fixed_throttle_value=val)

        elif op == "mode":
            # mode ai
            mode = cmd[1]
            current_car.update_settings(throttle_mode=mode)
            
        elif op == "exit":
            break

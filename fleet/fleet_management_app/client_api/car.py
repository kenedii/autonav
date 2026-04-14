import time
import threading
import logging
import numpy as np
import os
import math
from typing import List, Dict, Any

try:
    from .hardware import PicoSerialController, get_camera, get_system_specs
    from .models import AutonomousDriver, ObjectDetector
    from .slam import VisualSlamSystem
except ImportError:
    from hardware import PicoSerialController, get_camera, get_system_specs
    from models import AutonomousDriver, ObjectDetector
    from slam import VisualSlamSystem

logger = logging.getLogger("CarClient")

class CarClient:
    def __init__(self):
        self.config = {}
        self.running = False
        self.paused = False
        self.pause_until = 0
        self.state = {
            "location": None, 
            "last_action": None, 
            "fps": 0,
            "detections": [],
            "specs": {}
        }
        
        # Get immediate specs
        try:
            self.state["specs"] = get_system_specs() 
        except Exception as e:
            logger.warning(f"Failed to get system specs: {e}")

        self.camera = None
        self.motor_controller = None
        self.control_model = None
        self.detection_model = None
        self.slam = None
        self.action_loop = ['control', 'api']
        
        self.target_dest = None # (x, y)
        self.nav_kp = 2.0 # Proportional gain for steering
        
        self.thread = None
        self.lock = threading.Lock()
        
        # Hardware Constants
        self.STEERING_CHANNEL = 0
        self.THROTTLE_CHANNEL = 1
        self.STEERING_CENTER = 1500
        self.THROTTLE_CENTER = 1500
        self.THROTTLE_MAX = 1900
        self.THROTTLE_MIN = 1200
        
        # Default Throttle
        self.fixed_throttle = 0.22
        
        # Try Auto-Config (Defaults)
        import os
        if os.path.exists("best_model.pth"):
             logger.info("Auto-loading local 'best_model.pth'...")
             default_config = {
                 "device": "cuda",
                 "architecture": "resnet101",
                 "cameras": [{"type": "realsense", "width": 640, "height": 480, "fps": 15}],
                 "control_model_type": "pytorch",
                 "control_model": "best_model.pth", 
                 "detection_model": "yolov8n.pt", 
                 "action_loop": ["control", "detection"]
             }
             
             # Check for optimized TRT model
             trt_path = "/home/jetson/jetracer_run/checkpoints/checkpoints/model_7_resnet101/best_model_trt.pth"
             if os.path.exists(trt_path):
                 logger.info(f"Found optimized TensorRT model at {trt_path}")
                 default_config["control_model_type"] = "tensorrt"
                 default_config["control_model"] = trt_path
             try:
                 self.configure(default_config)
             except Exception as e:
                 logger.error(f"Auto-config failed: {e}")

    def set_throttle_mode(self, mode, value=None):
        with self.lock:
             self.config['throttle_mode'] = mode
             if value is not None:
                 self.fixed_throttle = value
             logger.info(f"Throttle mode set to {mode} (val={value})")

    def configure(self, config: Dict[str, Any]):
        with self.lock:
            # Expand ~ in paths
            if config.get("control_model"):
                config["control_model"] = os.path.expanduser(config["control_model"])
                # Remove "jetson:" prefix if user included it by mistake
                if config["control_model"].startswith("jetson:"):
                    config["control_model"] = config["control_model"].replace("jetson:", "")
            if config.get("detection_model"):
                config["detection_model"] = os.path.expanduser(config["detection_model"])

            # Set action loop first before initializing hardware/models
            self.action_loop = config.get("action_loop", ["control", "api"])
            
            # --- Update Specs from Config ---
            try:
                specs = get_system_specs(config.get("cameras", []))
                if config.get("architecture"): 
                    specs["resnet_version"] = config["architecture"]
                if config.get("detection_model"): 
                    specs["yolo_version"] = os.path.basename(config["detection_model"])
                self.state["specs"] = specs
            except Exception as e:
                logger.warning(f"Error updating specs: {e}")

            # Re-init Camera if needed
            if "cameras" in config:
                if self.camera:
                    self.camera.release()
                    self.camera = None
            
            # Stop existing but don't join thread if we are called FROM the thread (avoid deadlock)
            self.running = False 
            if self.camera:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            
            self.config = config
            logger.info(f"Configuring CarClient with model: {config.get('control_model')}...")
            
            # Setup Hardware
            try:
                if self.motor_controller is None:
                    self.motor_controller = PicoSerialController()
                self.motor_controller.set_us(self.STEERING_CHANNEL, self.STEERING_CENTER)
                self.motor_controller.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
            except Exception as e:
                logger.error(f"Failed to init Pico serial motor controller (Mocking): {e}")
                self.motor_controller = None 
                
            try:
                # Setup Camera (first only)
                if config.get("cameras"):
                    cam_conf = config["cameras"][0]
                    # Check if any module needs depth
                    need_depth = 'detection' in self.action_loop or 'slam' in self.action_loop
                    self.camera = get_camera(cam_conf, enable_depth=need_depth)
                
                if self.camera is None:
                     logger.warning("Camera initialization returned None")
            except Exception as e:
                logger.error(f"Failed to init Camera: {e}")
                self.camera = None 

            # Pre-arm models
            try:
                # Setup Models
                self.control_model = AutonomousDriver(config)
                # Only load detection if in loop to save memory (detection is heavy)
                if 'detection' in self.action_loop:
                    self.detection_model = ObjectDetector(config)
                else:
                    self.detection_model = None
                
                # Setup SLAM
                if 'slam' in self.action_loop:
                    w = config.get("cameras", [{}])[0].get("width", 640)
                    h = config.get("cameras", [{}])[0].get("height", 480)
                    self.slam = VisualSlamSystem(width=w, height=h)
                else:
                    self.slam = None

            except Exception as e:
                logger.error(f"Failed to init Models: {e}")
                self.control_model = None
                self.detection_model = None
                self.slam = None

    def start_logic(self):
        # Ensure we have a camera and models, or at least a camera to start the loop
        if self.running: 
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Car logic started.")

    def stop_logic(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Safe stop
        if self.motor_controller:
            self.motor_controller.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
            self.motor_controller.set_us(self.STEERING_CHANNEL, self.STEERING_CENTER)
            self.motor_controller.close()
            self.motor_controller = None
            
        if self.camera:
            self.camera.release()
            self.camera = None

    def pause(self, duration=None):
        """Pause execution. If duration is set, auto-resume after seconds."""
        self.paused = True
        if self.motor_controller:
            self.motor_controller.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
        
        if duration:
            self.pause_until = time.time() + duration
            logger.info(f"Paused for {duration} seconds")
        else:
            self.pause_until = 0 # Indefinite
            logger.info("Paused indefinitely")

    def resume(self):
        self.paused = False
        self.pause_until = 0
        logger.info("Resumed")

    def _loop(self):
        frame_count = 0
        last_time = time.time()
        
        # Arm ESC
        if self.motor_controller:
             time.sleep(1.0)
        
        throttle_us = self.THROTTLE_CENTER + int(self.fixed_throttle * (self.THROTTLE_MAX - self.THROTTLE_CENTER))
        
        while self.running:
            loop_start = time.time()
            
            # Check pause state
            if self.paused:
                if self.pause_until > 0 and time.time() > self.pause_until:
                    self.resume()
                else:
                    time.sleep(0.1)
                    continue

            # Get Frame
            frame_color, frame_depth, imu_data = None, None, None
            if self.camera:
                frame_color, frame_depth, imu_data = self.camera.read()

            if frame_color is None:
                if not self.camera:
                    logger.warning("No camera configured! Waiting for configuration...")
                    time.sleep(2.0)
                else:
                    time.sleep(0.01)
                continue

            # Execute Action Loop
            steer_val = 0.0
            
            # Determine Throttle
            throttle_mode = self.config.get('throttle_mode', 'fixed')
            current_throttle = self.fixed_throttle
            
            # --- SLAM & Navigation ---
            override_steer = None
            if self.slam and frame_color is not None:
                # Update Pose
                pose = self.slam.update(frame_color, frame_depth, throttle_val=current_throttle, imu_data=imu_data)
                self.state['location'] = pose
                
                # Navigation Logic
                if self.target_dest:
                    dx = self.target_dest[0] - pose['x']
                    dy = self.target_dest[1] - pose['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 0.2: # Arrived (20cm radius)
                        logger.info(f"Nav: Reached destination {self.target_dest}")
                        self.target_dest = None
                        override_steer = 0.0
                        # Optional: Stop car?
                        # current_throttle = 0.0
                    else:
                        # P-Control for Heading
                        target_theta = math.atan2(dy, dx)
                        error = target_theta - pose['theta']
                        # Normalize angle [-pi, pi]
                        error = (error + math.pi) % (2 * math.pi) - math.pi
                        
                        # Apply Gain
                        override_steer = np.clip(error * self.nav_kp, -1.0, 1.0)
                        # Maybe slow down if turning hard?
                        # if abs(override_steer) > 0.5: current_throttle *= 0.8

            for action in self.action_loop:
                if action == 'control':
                    # Run Control Model
                    if self.control_model is None:
                        # logger.warning("Control model not initialized!")
                        continue
                    
                    if override_steer is not None:
                        steer_norm = override_steer
                    else:
                        steer_norm = self.control_model.predict(frame_color)
                    
                    if throttle_mode == 'ai':
                        # Placeholder: if model returned throttle, we would use it here.
                        pass 

                    # Convert to PWM
                    # steer_norm is -1 to 1
                    # 1500 center. 2000 max (right), 1000 min (left)
                    # Assuming +1 is Right, -1 is Left
                    pulse = int(self.STEERING_CENTER + (steer_norm * 500))
                    pulse = np.clip(pulse, 1000, 2000)
                    
                    # Calculate throttle pulse
                    throttle_us = self.THROTTLE_CENTER + int(current_throttle * (self.THROTTLE_MAX - self.THROTTLE_CENTER))
                    
                    if self.motor_controller:
                        self.motor_controller.set_us(self.STEERING_CHANNEL, pulse)
                        self.motor_controller.set_us(self.THROTTLE_CHANNEL, throttle_us)
                        
                    self.state["last_action"] = {"steer": steer_norm, "throttle": current_throttle}
                    
                elif action == 'detection':
                    # Run Detection
                    if self.detection_model is None:
                        continue
                        
                    detections = self.detection_model.detect(frame_color)
                    
                    # Calculate depth if available
                    if frame_depth is not None:
                        for d in detections:
                            bbox = d['bbox'] # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, bbox)
                            # Clip to frame
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(frame_depth.shape[1], x2); y2 = min(frame_depth.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                crop = frame_depth[y1:y2, x1:x2]
                                # Filter out zero (invalid) depth
                                valid_depths = crop[crop > 0]
                                if len(valid_depths) > 0:
                                    # Depth is in unknown units, usually mm or meters depending on config
                                    # RS usually mm.
                                    avg_dist = np.mean(valid_depths)
                                    d['distance'] = float(avg_dist)
                    
                    self.state["detections"] = detections
                    
                elif action == 'api':
                    # Update state variables that API reads
                    pass
            
            # FPS Calculation
            frame_count += 1
            if time.time() - last_time > 1.0:
                self.state["fps"] = frame_count
                frame_count = 0
                last_time = time.time()
            
            # Optional: yield slightly to not hog CPU if loop is extremely fast
            time.sleep(0.001)

car = CarClient()

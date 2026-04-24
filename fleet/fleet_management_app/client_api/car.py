import time
import threading
import logging
import numpy as np
import os
import math
from typing import List, Dict, Any

try:
    from .hardware import get_camera, get_system_specs, get_motor_controller
    from .models import AutonomousDriver, ObjectDetector
    from .slam import VisualSlamSystem
except ImportError:
    from hardware import get_camera, get_system_specs, get_motor_controller
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
            "detection_count": 0,
            "yolo_enabled": False,
            "slam_enabled": False,
            "slam_map_points": 0,
            "navigation": {"active": False, "target": None},
            "specs": {}
        }
        self.detection_history = []
        self.max_detection_history = 1000
        
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
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        autonav_v2_path = os.path.join(
            repo_root,
            "checkpoints",
            "AutoNav-v2",
            "AutoNav-v2-34",
            "AutoNav-v2-34.pth",
        )
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
        elif os.path.exists(autonav_v2_path):
             logger.info("Auto-loading local AutoNav v2 checkpoint at %s", autonav_v2_path)
             default_config = {
                 "device": "cuda",
                 "architecture": "resnet34",
                 "cameras": [{"type": "realsense", "width": 640, "height": 480, "fps": 15}],
                 "control_model_type": "pytorch",
                 "control_model": autonav_v2_path,
                 "detection_model": "yolov8n.pt",
                 "throttle_mode": "ai",
                 "invert_steering": True,
                 "throttle_output_scale": 3.33,
                 "action_loop": ["control", "detection"],
             }
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

    @staticmethod
    def _build_action_loop(config):
        base = ["control", "api"]
        yolo_enabled = bool(config.get("yolo_enabled", "detection" in config.get("action_loop", [])))
        slam_enabled = bool(config.get("slam_enabled", "slam" in config.get("action_loop", [])))
        if yolo_enabled:
            base.insert(1, "detection")
        if slam_enabled:
            base.insert(1, "slam")
        return base

    @staticmethod
    def _extract_control_prediction(prediction):
        if isinstance(prediction, dict):
            steer_norm = float(prediction.get("steering", 0.0))
            throttle = prediction.get("throttle")
            return steer_norm, None if throttle is None else float(throttle)

        values = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return 0.0, None
        if values.size == 1:
            return float(values[0]), None
        return float(values[0]), float(values[1])

    def _throttle_to_pulse_width(self, throttle_norm):
        throttle_norm = float(np.clip(throttle_norm, -1.0, 1.0))
        if throttle_norm >= 0.0:
            span = self.THROTTLE_MAX - self.THROTTLE_CENTER
        else:
            span = self.THROTTLE_CENTER - self.THROTTLE_MIN
        return int(self.THROTTLE_CENTER + (throttle_norm * span))

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
            config.setdefault("controller_backend", "pico")
            config.setdefault("device", "cuda")
            if config.get("nav_kp") is not None:
                self.nav_kp = float(config.get("nav_kp"))

            if config.get("pca_address") is None:
                config["pca_address"] = 0x40

            # Set action loop first before initializing hardware/models
            self.action_loop = self._build_action_loop(config)
            config["action_loop"] = list(self.action_loop)
            config["yolo_enabled"] = "detection" in self.action_loop
            config["slam_enabled"] = "slam" in self.action_loop
            
            # --- Update Specs from Config ---
            try:
                specs = get_system_specs(config.get("cameras", []), config=config)
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
                if self.motor_controller:
                    try:
                        self.motor_controller.close()
                    except Exception:
                        pass
                self.motor_controller = get_motor_controller(config)
                self.motor_controller.set_us(self.STEERING_CHANNEL, self.STEERING_CENTER)
                self.motor_controller.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
            except Exception as e:
                logger.error(f"Failed to init motor controller (Mocking): {e}")
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
                    self.state['location'] = None

                self.state["yolo_enabled"] = self.detection_model is not None
                self.state["slam_enabled"] = self.slam is not None
                if self.slam is not None:
                    self.state["slam_map_points"] = len(self.slam.trajectory)
                else:
                    self.state["slam_map_points"] = 0

            except Exception as e:
                logger.error(f"Failed to init Models: {e}")
                self.control_model = None
                self.detection_model = None
                self.slam = None

    def set_navigation_target(self, x, y):
        self.target_dest = (float(x), float(y))
        self.state["navigation"] = {"active": True, "target": {"x": float(x), "y": float(y)}}

    def cancel_navigation(self):
        self.target_dest = None
        self.state["navigation"] = {"active": False, "target": None}

    def reset_slam(self):
        if self.slam is None:
            return False
        self.slam.reset()
        self.state["location"] = self.slam._get_state()
        self.state["slam_map_points"] = 0
        return True

    def get_slam_map(self):
        location = self.state.get("location") or {}
        return {
            "location": location,
            "trajectory": location.get("trajectory", []),
            "target": self.target_dest,
            "active": self.slam is not None,
        }

    def get_detection_history(self, limit=200):
        limit = max(1, min(int(limit), self.max_detection_history))
        return self.detection_history[-limit:]

    def clear_detection_history(self):
        self.detection_history = []

    def configure_yolo(self, enabled=None, model_path=None, conf_threshold=None, iou_threshold=None, max_detections=None):
        was_running = self.running
        updated = dict(self.config)
        if enabled is not None:
            updated["yolo_enabled"] = bool(enabled)
        if model_path:
            updated["detection_model"] = model_path
        if conf_threshold is not None:
            updated["yolo_confidence_threshold"] = float(conf_threshold)
        if iou_threshold is not None:
            updated["yolo_iou_threshold"] = float(iou_threshold)
        if max_detections is not None:
            updated["yolo_max_detections"] = int(max_detections)
        self.configure(updated)
        if was_running and not self.running:
            self.start_logic()
        return {
            "yolo_enabled": self.state.get("yolo_enabled", False),
            "detection_model": self.config.get("detection_model"),
            "thresholds": {
                "confidence": self.config.get("yolo_confidence_threshold", 0.25),
                "iou": self.config.get("yolo_iou_threshold", 0.45),
                "max_detections": self.config.get("yolo_max_detections", 100),
            },
        }

    def configure_slam(self, enabled=None, nav_kp=None):
        was_running = self.running
        updated = dict(self.config)
        if enabled is not None:
            updated["slam_enabled"] = bool(enabled)
        if nav_kp is not None:
            self.nav_kp = float(nav_kp)
        self.configure(updated)
        if was_running and not self.running:
            self.start_logic()
        return {
            "slam_enabled": self.state.get("slam_enabled", False),
            "nav_kp": self.nav_kp,
        }

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
        
        throttle_us = self._throttle_to_pulse_width(self.fixed_throttle)
        
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
                        self.state["navigation"] = {"active": False, "target": None}
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
                        self.state["navigation"] = {
                            "active": True,
                            "target": {"x": float(self.target_dest[0]), "y": float(self.target_dest[1])},
                            "distance_m": float(dist),
                            "heading_error_rad": float(error),
                        }
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
                        ai_throttle = None
                    else:
                        prediction = self.control_model.predict(frame_color)
                        steer_norm, ai_throttle = self._extract_control_prediction(prediction)

                    if throttle_mode == 'ai' and ai_throttle is not None:
                        current_throttle = ai_throttle

                    # Convert to PWM
                    # steer_norm is -1 to 1
                    # 1500 center. 2000 max (right), 1000 min (left)
                    # Assuming +1 is Right, -1 is Left
                    pulse = int(self.STEERING_CENTER + (steer_norm * 500))
                    pulse = np.clip(pulse, 1000, 2000)
                    
                    # Calculate throttle pulse
                    throttle_us = self._throttle_to_pulse_width(current_throttle)
                    
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
                    self.state["detection_count"] = len(detections)
                    timestamp = time.time()
                    history_entry = {
                        "timestamp": timestamp,
                        "count": len(detections),
                        "detections": detections,
                    }
                    self.detection_history.append(history_entry)
                    if len(self.detection_history) > self.max_detection_history:
                        self.detection_history = self.detection_history[-self.max_detection_history:]
                    
                elif action == 'api':
                    # Update state variables that API reads
                    pass

            self.state["yolo_enabled"] = self.detection_model is not None
            self.state["slam_enabled"] = self.slam is not None
            if self.slam and self.state.get("location"):
                self.state["slam_map_points"] = len(self.state["location"].get("trajectory", []))
            else:
                self.state["slam_map_points"] = 0
            
            # FPS Calculation
            frame_count += 1
            if time.time() - last_time > 1.0:
                self.state["fps"] = frame_count
                frame_count = 0
                last_time = time.time()
            
            # Optional: yield slightly to not hog CPU if loop is extremely fast
            time.sleep(0.001)

car = CarClient()

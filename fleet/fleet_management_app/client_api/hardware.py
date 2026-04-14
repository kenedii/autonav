import cv2
import numpy as np
import time
import sys
import platform
import subprocess
import os

try:
    import serial
except ImportError:
    serial = None

# Try importing pyrealsense2, handle if missing
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

class PicoSerialController:
    """Send steering/throttle PWM pulse widths to a Raspberry Pi Pico over serial."""

    def __init__(self, port=None, baudrate=115200, timeout=0.2):
        if serial is None:
            raise RuntimeError("pyserial is required for Pico serial control")

        self.port = port or os.getenv("PICO_SERIAL_PORT", "/dev/ttyACM0")
        self.baudrate = int(os.getenv("PICO_SERIAL_BAUD", str(baudrate)))
        self.timeout = timeout

        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )

        # Let USB CDC settle on connect and set a neutral state.
        time.sleep(1.0)
        self.set_us(0, 1500)
        self.set_us(1, 1500)

    def _send_line(self, cmd):
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

    def set_us(self, channel, microseconds):
        us = int(max(500, min(2500, microseconds)))
        self._send_line(f"SET {int(channel)} {us}")

    def close(self):
        try:
            if getattr(self, "ser", None) and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

def get_cpu_ram_info():
    try:
        # Simple linux commands
        cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode().strip().split(': ')[1]
    except:
        cpu_info = platform.machine() # fallback
    
    try:
        mem_info = subprocess.check_output("free -h | grep Mem | awk '{print $2}'", shell=True).decode().strip()
    except:
        mem_info = "Unknown"
        
    return f"{cpu_info} / {mem_info}"

def get_system_specs(cameras=[]):
    """
    Gather hardware specs.
    cameras: list of camera objects or configs
    """
    specs = {
        "device": f"{platform.system()} {platform.release()}",
        "cpu_ram": get_cpu_ram_info(),
        "cameras": [],
        "inference": "CUDA (GPU)" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "CPU",
        "resnet_version": "ResNet-101 (Default)",
        "yolo_version": "YOLOv8n (Default)"
    }
    
    # Process cameras
    # This expects the 'cameras' arg to be the list of CameraConfig or actual Camera objects
    # For now, let's just describe what's passed or check connected devices
    # But since main.py manages the camera objects, we can just pass the config.
    
    return specs

class CameraInterface:
    def read(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

class OpenCVCamera(CameraInterface):
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        
        # Try GStreamer pipeline for Jetson CSI Camera first if index is 0
        if index == 0:
            pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                f"width=(int){width}, height=(int){height}, "
                f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                print("[Camera] GStreamer failed, falling back to V4L2...")
                self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(index)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open OpenCV camera index {index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None
        
        # Ensure correct size (sometimes GStreamer or V4L2 returns different sizes)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # Convert BGR to RGB, return None for depth and imu
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, None

    def release(self):
        self.cap.release()

class RealSenseCamera(CameraInterface):
    def __init__(self, width=640, height=480, fps=15, enable_depth=True):
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 not installed")
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # RealSense supports specific FPS: 6, 15, 30, 60.
        # Mapping to closest supported.
        if fps > 45: r_fps = 60
        elif fps > 22: r_fps = 30
        elif fps > 10: r_fps = 15
        else: r_fps = 6

        print(f"[Camera] RealSense requesting {width}x{height}@{r_fps} FPS (Depth: {enable_depth})")
        
        try:
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, r_fps)
            if enable_depth:
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, r_fps)
            
            # Enable IMU if available (D435i/D455)
            try:
                config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63) # 63 or 250 Hz
                config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200) # 200 or 400 Hz
                print("[Camera] Enabling IMU streams (Accel/Gyro)")
            except Exception as imu_e:
                print(f"[Camera] IMU not available or failed: {imu_e}")

            self.profile = self.pipeline.start(config)
            
            # Optimization: Turn off laser if we don't need depth (saves power/heat)
            if not enable_depth:
                try:
                    dev = self.profile.get_device()
                    depth_sensor = dev.first_depth_sensor()
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 0)
                except: pass

        except Exception as e:
            print(f"[Camera] Profile fail: {e}. Trying fallback...")
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
            # Only fallback to depth if enabled
            if enable_depth:
                try:
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                except: pass
            self.profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color) if enable_depth else None
        self.depth_enabled = enable_depth
        
        # Power management: Set auto-exposure for stable FPS
        try:
            sensors = self.profile.get_device().query_sensors()
            for s in sensors:
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 1)
        except: pass
            
    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            
            color_frame = frames.get_color_frame()
            depth_data = None
            imu_data = {}

            # Get IMU data (accelerometer + gyro)
            try:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if accel_frame:
                    val = accel_frame.as_motion_frame().get_motion_data()
                    imu_data['accel'] = [val.x, val.y, val.z]
                if gyro_frame:
                    val = gyro_frame.as_motion_frame().get_motion_data()
                    imu_data['gyro'] = [val.x, val.y, val.z]
            except Exception:
                pass # IMU frame not available this cycle

            if self.depth_enabled and self.align:
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                if depth_frame:
                    depth_data = np.asanyarray(depth_frame.get_data())
            
            if not color_frame:
                return None, None, None
                
            color_data = np.asanyarray(color_frame.get_data())
            return color_data, depth_data, imu_data
        except Exception as e:
            # Don't print every frame to avoid log spam, just yield
            return None, None, None

    def release(self):
        try:
            self.pipeline.stop()
            print("[Camera] RealSense pipeline stopped.")
            time.sleep(0.5)
        except:
            pass

def get_camera(config, enable_depth=True):
    # config example: {"type": "realsense", "width": 848, "height": 480, "fps": 30}
    c_type = config.get("type", "opencv").lower()
    print(f"[Camera] Initializing {c_type} (Need Depth: {enable_depth})...")
    if c_type == "realsense":
        return RealSenseCamera(
            width=config.get("width", 640),
            height=config.get("height", 480),
            fps=config.get("fps", 15),
            enable_depth=enable_depth
        )
    else:
        return OpenCVCamera(
            index=config.get("index", 0),
            width=config.get("width", 640),
            height=config.get("height", 480),
            fps=config.get("fps", 30)
        )
        return OpenCVCamera(
            index=config.get("index", 0),
            width=config.get("width", 640),
            height=config.get("height", 480),
            fps=config.get("fps", 30)
        )

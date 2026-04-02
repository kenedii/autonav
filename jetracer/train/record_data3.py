#!/usr/bin/env python3
"""
Minimal RC car data collection with session management:
- Uses only RGB camera and a single front depth value (via realsense_full)
- Immediate PWM output via PCA9685 (smbus2)
- Supports multi-session recording and deletion of last N frames or entire session
- Throttle capped at configurable percent of full speed
- All 3 control modes supported (sticks + triggers)
"""

import os
import csv
import time
import threading
import atexit
import queue
from datetime import datetime
import socket
import json
import pygame
from smbus2 import SMBus
import cv2
import realsense_full  # RealSense pipeline
import numpy as np
import select
import argparse
import queue
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from preprocess_utils import (
    apply_preprocess_profile,
    LEGACY_PREPROCESS_PROFILE,
    infer_preprocess_profile,
    PREPROCESS_OUTPUT_HEIGHT,
    PREPROCESS_OUTPUT_WIDTH,
)

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    GST_AVAILABLE = True
except Exception:
    Gst = None
    GST_AVAILABLE = False

# ================= ARGS =================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "opencv", "other"], help="Camera type")
parser.add_argument("--device", type=int, default=0, help="Camera device ID (for opencv)")
parser.add_argument('--primary_rgb_source', type=str, default='main_camera', choices=['main_camera', 'cam0'], help='Primary RGB source for canonical rgb_path')
parser.add_argument('--record_mode', type=str, default='rgb', choices=['rgb', 'all'], help='Recording mode: "rgb" for control+RGB only, "all" for control+RGB+IR+depth')
parser.add_argument('--control_mode', type=str, default=None, choices=['joystick', 'steer_trigger'], help='Optional override for control mapping: "steer_trigger" uses left trigger for accel')
parser.add_argument('--always_save', action='store_true', help='Save frames at TARGET_FPS even when controls have not changed')
parser.add_argument('--view_360', action='store_true', help='Also record two Jetson CSI cameras connected to CAM0/CAM1')
parser.add_argument('--view_360_cam0_sensor_id', type=int, default=0, help='Jetson CSI sensor-id for the CAM0 360 camera')
parser.add_argument('--view_360_cam1_sensor_id', type=int, default=1, help='Jetson CSI sensor-id for the CAM1 360 camera')
parser.add_argument('--view_360_width', type=int, default=640, help='Capture width for the 360 cameras')
parser.add_argument('--view_360_height', type=int, default=480, help='Capture height for the 360 cameras')
parser.add_argument('--view_360_fps', type=int, default=15, help='Capture FPS for the 360 cameras')
parser.add_argument('--view_360_save_width', type=int, default=320, help='Saved width for the 360 camera images')
parser.add_argument('--view_360_save_height', type=int, default=240, help='Saved height for the 360 camera images')
parser.add_argument('--view_360_cam0_flip_method', type=int, default=2, help='nvvidconv flip-method for CAM0 (0-7)')
parser.add_argument('--view_360_cam1_flip_method', type=int, default=2, help='nvvidconv flip-method for CAM1 (0-7)')
args = parser.parse_args()

PRIMARY_RGB_SOURCE = args.primary_rgb_source
PRIMARY_RGB_IS_CAM0 = PRIMARY_RGB_SOURCE == "cam0"
PREPROCESS_PROFILE = infer_preprocess_profile(
    camera_configs=[
        {"role": "primary_rgb", "type": "csi"} if PRIMARY_RGB_IS_CAM0 else {"role": "primary_rgb", "type": args.camera}
    ],
)
DEFAULT_OUTPUT_SIZE = (PREPROCESS_OUTPUT_WIDTH, PREPROCESS_OUTPUT_HEIGHT)
RUN_ID = None

if args.camera in ["opencv", "other"]:
    realsense_full.set_camera_type("opencv", args.device)
else:
    realsense_full.set_camera_type("realsense")

# ================= INPUT MODE =================
# Switch between local Xbox (direct USB/pygame) vs. network controller (from laptop)
USE_NETWORK_CONTROLLER = True
NET_HOST = "0.0.0.0"
NET_PORT = 5007
# Hotspot-friendly timeout: allow brief jitter before neutralizing
NET_TIMEOUT_S = 1.0

# ================= DEBUG/DIAG FLAGS =================
# Set to False to bypass camera during recording (for lag diagnostics)
CAMERA_ENABLED = True

# ================= RAM BUFFER =================
# Stores tuples of (rgb_image, row_data)
ram_buffer = []

# ================= CONFIG =================
class Config:
    PCA_ADDR = 0x40
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1

    STEERING_AXIS = 0
    THROTTLE_AXIS = 1        # Default to left Y, will be overridden if mode 2
    RIGHT_THROTTLE_AXIS = 4  # Right stick Y for separate mode
    LEFT_TRIGGER_AXIS = 2    # LT (original working)
    RIGHT_TRIGGER_AXIS = 5   # RT (original working)

    PWM_FREQ = 50
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    TARGET_FPS = 10
    DELETE_N_FRAMES = 12

    # Safety scaling (set to safe defaults; change if needed)
    THROTTLE_MAX_SCALE = 0.30  # 30% of full travel
    STEERING_MAX_SCALE = 1.00  # 100% steering
    
    # Steering Gamma for fine control (1.0 = linear, 2.0 = quadratic)
    STEERING_GAMMA = 2.5

    # Deadzone to ignore tiny stick/trigger noise
    AXIS_DEADZONE = 0.03

cfg = Config()


class CsiCameraStream:
    """Low-latency Jetson CSI camera reader backed by nvarguscamerasrc."""

    def __init__(self, sensor_id, width, height, fps, flip_method=0, label="CAM"):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.label = label
        self.cap = None
        self.pipeline = None
        self.appsink = None
        self.backend = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    @staticmethod
    def build_pipeline(sensor_id, width, height, fps, flip_method, appsink_name=None):
        appsink = "appsink drop=true max-buffers=1 sync=false"
        if appsink_name is not None:
            appsink = (
                f"appsink name={appsink_name} emit-signals=false "
                "drop=true max-buffers=1 sync=false"
            )

        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            f"{appsink}"
        )

    def start(self):
        if self.cap is not None or self.pipeline is not None:
            return True

        self.stop_event.clear()
        started = False

        if GST_AVAILABLE:
            started = self._start_native_gstreamer()

        if not started:
            started = self._start_opencv_gstreamer()

        if not started:
            print(f"[360] Failed to open {self.label} on sensor-id={self.sensor_id}")
            return False

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        print(
            f"[360] {self.label} started on sensor-id={self.sensor_id} using {self.backend} "
            f"({self.width}x{self.height}@{self.fps}, flip={self.flip_method})"
        )
        return True

    def _start_native_gstreamer(self):
        try:
            pipeline_desc = self.build_pipeline(
                self.sensor_id,
                self.width,
                self.height,
                self.fps,
                self.flip_method,
                appsink_name="capture_sink",
            )
            self.pipeline = Gst.parse_launch(pipeline_desc)
            self.appsink = self.pipeline.get_by_name("capture_sink")
            if self.appsink is None:
                raise RuntimeError("appsink not found in GStreamer pipeline")

            state_ret = self.pipeline.set_state(Gst.State.PLAYING)
            if state_ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("pipeline failed to enter PLAYING state")

            self.backend = "python-gstreamer"
            return True
        except Exception as e:
            print(f"[360] {self.label} native GStreamer open failed: {e}")
            if self.pipeline is not None:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            self.pipeline = None
            self.appsink = None
            self.backend = None
            return False

    def _start_opencv_gstreamer(self):
        pipeline = self.build_pipeline(
            self.sensor_id,
            self.width,
            self.height,
            self.fps,
            self.flip_method,
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            return False

        self.backend = "opencv-gstreamer"
        return True

    def _pull_gstreamer_sample(self):
        if self.appsink is None:
            return None

        sample = self.appsink.emit("try-pull-sample", 200000000)
        if sample is None:
            return None

        return self._sample_to_bgr(sample)

    @staticmethod
    def _sample_to_bgr(sample):
        caps = sample.get_caps()
        if caps is None or caps.get_size() == 0:
            return None

        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        pixel_format = structure.get_value("format")

        buffer = sample.get_buffer()
        if buffer is None:
            return None

        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return None

        try:
            channels = 3 if pixel_format == "BGR" else 4 if pixel_format == "BGRx" else None
            if channels is None:
                return None

            data = np.frombuffer(map_info.data, dtype=np.uint8)
            row_stride = data.size // height if height else 0
            if row_stride < width * channels:
                return None

            frame = data.reshape((height, row_stride))
            frame = frame[:, : width * channels]
            frame = frame.reshape((height, width, channels))
            if channels == 4:
                frame = frame[:, :, :3]
            return frame.copy()
        finally:
            buffer.unmap(map_info)

    def _reader(self):
        while not self.stop_event.is_set():
            if self.backend == "python-gstreamer":
                frame = self._pull_gstreamer_sample()
            else:
                if self.cap is None:
                    break
                ok, frame = self.cap.read()
                if not ok:
                    frame = None

            if frame is None:
                time.sleep(0.02)
                continue

            with self.frame_lock:
                self.latest_frame = frame.copy()

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None
        self.appsink = None
        self.backend = None
        self.thread = None


class CsiRigCapture:
    def __init__(self, enable_cam0=True, enable_cam1=True):
        self.enable_cam0 = enable_cam0
        self.enable_cam1 = enable_cam1
        self.cam0 = None
        self.cam1 = None
        self.cam0_ok = False
        self.cam1_ok = False
        self.enabled = False

        if self.enable_cam0:
            self.cam0 = CsiCameraStream(
                sensor_id=args.view_360_cam0_sensor_id,
                width=args.view_360_width,
                height=args.view_360_height,
                fps=args.view_360_fps,
                flip_method=args.view_360_cam0_flip_method,
                label="CAM0",
            )
        if self.enable_cam1:
            self.cam1 = CsiCameraStream(
                sensor_id=args.view_360_cam1_sensor_id,
                width=args.view_360_width,
                height=args.view_360_height,
                fps=args.view_360_fps,
                flip_method=args.view_360_cam1_flip_method,
                label="CAM1",
            )

    def start(self):
        if self.cam0 is not None:
            self.cam0_ok = self.cam0.start()
        if self.cam1 is not None:
            self.cam1_ok = self.cam1.start()
        self.enabled = bool(self.cam0_ok or self.cam1_ok)
        if self.enable_cam0 and not self.cam0_ok:
            print("[360] CAM0 failed to start.")
        if self.enable_cam1 and not self.cam1_ok:
            print("[360] CAM1 failed to start.")
        return self.enabled

    def get_frames(self):
        cam0_frame = self.cam0.get_frame() if self.cam0 is not None else None
        cam1_frame = self.cam1.get_frame() if self.cam1 is not None else None
        return cam0_frame, cam1_frame

    def stop(self):
        if self.cam0 is not None:
            self.cam0.stop()
        if self.cam1 is not None:
            self.cam1.stop()
        self.enabled = False

# ================= PCA9685 (SMBus) =================
class PCA9685:
    def __init__(self, bus=1, address=cfg.PCA_ADDR):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(cfg.PWM_FREQ)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096 / freq_hz - 1
        prescale = int(prescaleval + 0.5)
        # reset + set prescale + restart
        self.bus.write_byte_data(self.address, 0x00, 0x10)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, (on >> 8) & 0xFF)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, (off >> 8) & 0xFF)

    def set_us(self, channel, microseconds):
        pulse_length = 1000000.0 / cfg.PWM_FREQ / 4096.0
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

pca = PCA9685()

# servo/esc pulse parameters (μs)
STEERING_CENTER = 1500
THROTTLE_CENTER = 1500
STEERING_MAX = 2000
STEERING_MIN = 1000
THROTTLE_MAX = 2000
THROTTLE_MIN = 1000

# Neutralize helper and safe exit
def neutralize():
    try:
        pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
        pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)
        # short delay to ensure ESC/servo sees neutral
        time.sleep(0.02)
    except Exception as e:
        print(f"[neutralize] PCA error: {e}")

# immediate neutral on startup and register for exit
neutralize()
atexit.register(neutralize)

# ================= PYGAME (conditional) =================
if not USE_NETWORK_CONTROLLER:
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("No joystick detected! Connect Xbox controller.")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")

# ================= CONTROL MODE SELECTION =================
print("\nChoose control mode:")
print("1: Left joystick only (horizontal: steer, vertical: throttle)")
print("2: Left joystick for steer (horizontal), Right joystick for throttle (vertical)")
print("3: Left joystick for steer (horizontal), Right trigger for accelerate, Left trigger for brake/reverse")
USE_LEFT_TRIGGER_ONLY = False
if args.control_mode == 'steer_trigger':
    # Non-interactive override: use steering on stick and left trigger for accelerate
    mode = 3
    USE_LEFT_TRIGGER_ONLY = True
    print("Selected control_mode=steer_trigger: steering by stick, left trigger for accel")
else:
    while True:
        mode_choice = input("Enter 1, 2, or 3: ").strip()
        if mode_choice == '1':
            cfg.THROTTLE_AXIS = cfg.THROTTLE_AXIS
            mode = 1
            print("Selected Mode 1: Left joystick controls both steering and throttle.")
            break
        elif mode_choice == '2':
            cfg.THROTTLE_AXIS = cfg.RIGHT_THROTTLE_AXIS
            mode = 2
            print("Selected Mode 2: Left joystick for steering, Right joystick for throttle.")
            break
        elif mode_choice == '3':
            mode = 3
            print("Selected Mode 3: Left joystick for steering, RT/LT for accelerate/brake (normalized).")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# ================= STEERING LIMIT SELECTION =================
steering_limit_input = input("\nEnter steering limit % (1-100, enter for 100): ").strip()
if steering_limit_input:
    try:
        pct = float(steering_limit_input)
        if 1 <= pct <= 100:
            cfg.STEERING_MAX_SCALE = pct / 100.0
            print(f"Steering limited to {pct}% of maximum.")
        else:
            print("Invalid range. Using 100%.")
    except ValueError:
        print("Invalid input. Using 100%.")
else:
    print("Using 100% steering.")

# ================= OPTIONAL 360 CSI CAMERAS =================
cam_rig = None
cam0_enabled = PRIMARY_RGB_IS_CAM0 or args.view_360
cam1_enabled = args.view_360
VIEW_360_ENABLED = False
view_360_waiting_logged = False

if (cam0_enabled or cam1_enabled) and CAMERA_ENABLED:
    cam_rig = CsiRigCapture(enable_cam0=cam0_enabled, enable_cam1=cam1_enabled)
    VIEW_360_ENABLED = cam_rig.start()
elif cam0_enabled or cam1_enabled:
    print("[360] CSI capture requested but CAMERA_ENABLED=False, so CAM0/CAM1 recording is disabled.")

# ================= MULTI-SESSION SETUP =================
BASE_RUN_DIR = "runs_rgb_depth"
os.makedirs(BASE_RUN_DIR, exist_ok=True)

def create_new_session():
    session_dir = os.path.join(BASE_RUN_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def get_dataset_header():
    return [
        "timestamp",
        "steer_us",
        "throttle_us",
        "steer_norm",
        "throttle_norm",
        "depth_front",
        "rgb_path",
        "rgb_source",
        "depth_source",
        "imu_source",
        "rear_rgb_source",
        "preprocess_profile",
        "run_id",
        "session_id",
        "cam0_path",
        "cam1_path",
        "ir_path",
        "depth_path",
    ]

RUN_DIR = create_new_session()
RUN_ID = os.path.basename(RUN_DIR)
csv_path = os.path.join(RUN_DIR, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(get_dataset_header())

frame_idx = 0

# ================= WRITER THREAD =================
write_queue = queue.Queue(maxsize=100)

def save_canonical_frame(image, path, profile, output_size=DEFAULT_OUTPUT_SIZE):
    if image is None or path is None:
        return

    if output_size == DEFAULT_OUTPUT_SIZE:
        image_small = apply_preprocess_profile(image, profile)
    else:
        image_small = cv2.resize(image, output_size)
    cv2.imwrite(path, image_small)

def build_row_metadata(rgb_source, depth_source, imu_source, rear_rgb_source, preprocess_profile, cam0_path, cam1_path, ir_path, depth_path):
    return [
        rgb_source,
        depth_source,
        imu_source,
        rear_rgb_source,
        preprocess_profile,
        RUN_ID,
        RUN_ID,
        cam0_path,
        cam1_path,
        ir_path,
        depth_path,
    ]

def writer_worker():
    global csv_file, writer
    while True:
        try:
            item = write_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if item is None:
            write_queue.task_done()
            break
            
        cmd = item[0]
        
        if cmd == "FRAME":
            _, rgb, row_data, rgb_path, rgb_profile, extra_frames = item
            try:
                save_canonical_frame(rgb, rgb_path, rgb_profile)
                for extra_path, extra_image in extra_frames:
                    save_canonical_frame(
                        extra_image,
                        extra_path,
                        LEGACY_PREPROCESS_PROFILE,
                        output_size=(args.view_360_save_width, args.view_360_save_height),
                    )
                writer.writerow(row_data)
                csv_file.flush()
            except Exception as e:
                print(f"Write error: {e}")
        elif cmd == "FRAME_ALL":
            _, rgb, ir_image, depth_map, row_data, rgb_path, rgb_profile, ir_path, depth_path, extra_frames = item
            try:
                save_canonical_frame(rgb, rgb_path, rgb_profile)
                if ir_image is not None and ir_path is not None:
                    cv2.imwrite(ir_path, ir_image)
                if depth_map is not None and depth_path is not None:
                    cv2.imwrite(depth_path, depth_map)
                for extra_path, extra_image in extra_frames:
                    save_canonical_frame(
                        extra_image,
                        extra_path,
                        LEGACY_PREPROCESS_PROFILE,
                        output_size=(args.view_360_save_width, args.view_360_save_height),
                    )
                writer.writerow(row_data)
                csv_file.flush()
            except Exception as e:
                print(f"Write error: {e}")
                
        elif cmd == "DELETE":
            n = item[1]
            try:
                csv_file.close()
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                # Keep header + (total - n) lines
                keep_count = max(1, len(lines) - n)
                with open(csv_path, 'w') as f:
                    f.writelines(lines[:keep_count])
                
                # Reopen
                csv_file = open(csv_path, "a", newline="")
                writer = csv.writer(csv_file)
                print(f"\n[Writer] Deleted last {n} entries from CSV.")
            except Exception as e:
                print(f"Delete error: {e}")
                
        write_queue.task_done()

threading.Thread(target=writer_worker, daemon=True).start()

# ================= HELPERS =================
def pwm_to_norm(us):
    return (us - 1500) / 500.0

def get_rgb_and_front_depth():
    if not CAMERA_ENABLED:
        return None, 0.0, None, None

    ir_image = None
    depth_map = None
    center_depth = 0.0
    rgb = None

    if PRIMARY_RGB_IS_CAM0:
        if cam_rig is None or not cam_rig.cam0_ok:
            return None, None, None, None

    if args.camera == 'realsense' and hasattr(realsense_full, 'get_all_frames') and args.record_mode == 'all':
        if hasattr(realsense_full, 'get_all_frames'):
            rgb, center_depth, ir_image, depth_map = realsense_full.get_all_frames()
        else:
            rgb, center_depth = realsense_full.get_aligned_frames()
        center_depth = float(center_depth or 0.0)
    else:
        rgb, center_depth = realsense_full.get_aligned_frames()
        center_depth = float(center_depth or 0.0)

    if PRIMARY_RGB_IS_CAM0 and cam_rig is not None:
        cam0_frame, _ = cam_rig.get_frames()
        rgb = cam0_frame

    if rgb is None:
        return None, None, None, None

    if not (args.record_mode == 'all' and args.camera == 'realsense'):
        ir_image = None
        depth_map = None

    return rgb, center_depth, ir_image, depth_map

def get_view_360_capture(frame_number):
    global view_360_waiting_logged

    if not VIEW_360_ENABLED or cam_rig is None:
        return [], []

    cam0_frame, cam1_frame = cam_rig.get_frames()
    extra_frames = []
    extra_paths = []

    if PRIMARY_RGB_IS_CAM0:
        if cam1_frame is not None:
            cam1_path = os.path.join(RUN_DIR, f"cam1_{frame_number:05d}.png")
            extra_frames.append((cam1_path, cam1_frame))
            extra_paths.append(cam1_path)
        elif cam1_enabled and not view_360_waiting_logged:
            print("\n[360] Waiting for CAM1 frames before saving rear-preview sidecar data...")
            view_360_waiting_logged = True
        view_360_waiting_logged = False if cam1_frame is not None else view_360_waiting_logged
        return extra_frames, extra_paths

    if cam0_frame is None and cam1_frame is None:
        if not view_360_waiting_logged:
            print("\n[360] Waiting for CAM0/CAM1 frames before saving synchronized 360 data...")
            view_360_waiting_logged = True
        return None, None

    if cam0_frame is not None:
        cam0_path = os.path.join(RUN_DIR, f"cam0_{frame_number:05d}.png")
        extra_frames.append((cam0_path, cam0_frame))
        extra_paths.append(cam0_path)
    if cam1_frame is not None:
        cam1_path = os.path.join(RUN_DIR, f"cam1_{frame_number:05d}.png")
        extra_frames.append((cam1_path, cam1_frame))
        extra_paths.append(cam1_path)

    view_360_waiting_logged = False
    return extra_frames, extra_paths

def delete_last_n(n):
    global frame_idx
    if frame_idx == 0:
        print("\nNothing to delete")
        return
    
    frame_idx = max(0, frame_idx - n)
    write_queue.put(("DELETE", n))
    print(f"\nRequested delete of last {n} frames -> index reverted to {frame_idx}")

def delete_current_session():
    global frame_idx, RUN_DIR, RUN_ID, csv_path, csv_file, writer, recording, view_360_waiting_logged
    confirm = input(f"\nDelete current session '{RUN_DIR}'? [y/N]: ").strip().lower()
    if confirm == 'y':
        recording = False
        print("\nDraining pending writes before deleting current session...")
        write_queue.join()

        csv_file.close()
        for fname in os.listdir(RUN_DIR):
            os.remove(os.path.join(RUN_DIR, fname))
        os.rmdir(RUN_DIR)
        print(f"Session '{RUN_DIR}' deleted!")
        RUN_DIR = create_new_session()
        RUN_ID = os.path.basename(RUN_DIR)
        csv_path = os.path.join(RUN_DIR, "dataset.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(get_dataset_header())
        view_360_waiting_logged = False
        frame_idx = 0


# ================= SHARED STATE =================
current_steer_us = STEERING_CENTER
current_throttle_us = THROTTLE_CENTER
recording = False
# recording_lock removed as we use queue for writing
# recording_lock = threading.Lock()

# Network controller state
net_last_ts = 0.0
net_steer_norm = 0.0
net_throttle_norm = 0.0

# ================= NETWORK LISTENER =================
def network_listener():
    global net_last_ts, net_steer_norm, net_throttle_norm
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((NET_HOST, NET_PORT))
    sock.setblocking(False)  # non-blocking so we can drain bursts without queueing
    print(f"[NET] Listening for controller on udp://{NET_HOST}:{NET_PORT}")

    recv_count = 0
    while True:
        try:
            # Wait up to 20ms for data, then drain everything available to keep only the latest
            ready, _, _ = select.select([sock], [], [], 0.02)
            if not ready:
                continue

            drained = 0
            last_msg = None
            while True:
                try:
                    data, _ = sock.recvfrom(256)
                    last_msg = data
                    drained += 1
                except BlockingIOError:
                    break

            if not last_msg:
                continue

            msg = json.loads(last_msg.decode('utf-8'))
            s = float(msg.get("s", 0.0))
            t = float(msg.get("t", 0.0))
            ts = float(msg.get("ts", time.time()))
            arrival_ts = time.time()
            # Clamp incoming values
            s = max(min(s, 1.0), -1.0)
            t = max(min(t, 1.0), -1.0)
            net_steer_norm = s
            net_throttle_norm = t
            # Use Jetson time for timeout logic to avoid clock skew with sender
            net_last_ts = arrival_ts
            recv_count += drained
            # print(f"[NET RX] #{recv_count} (drained {drained}) s={s:+.2f} t={t:+.2f} ts={ts:.2f} arrival={arrival_ts:.2f}")
        except Exception as e:
            print(f"[NET] recv error: {e}")

if USE_NETWORK_CONTROLLER:
    threading.Thread(target=network_listener, daemon=True).start()

# ================= RECORDING THREAD =================
def recording_worker():
    global frame_idx, recording, current_steer_us, current_throttle_us
    last_save_time = 0
    last_steer_rec = STEERING_CENTER
    last_throttle_rec = THROTTLE_CENTER
    MIN_CHANGE_US = 15

    print("Recording thread started...")
    
    while True:
        if recording:
            now = time.time()
            if now - last_save_time >= 1.0 / cfg.TARGET_FPS:
                # Check if inputs changed enough to warrant a frame (optional, but keeps dataset clean)
                # Access shared variables safely (integers are atomic, but good practice)
                s_us = current_steer_us
                t_us = current_throttle_us
                
                # Save if inputs changed beyond threshold OR user requested always-save
                if args.always_save or abs(s_us - last_steer_rec) >= MIN_CHANGE_US or abs(t_us - last_throttle_rec) >= MIN_CHANGE_US:
                        rgb, depth_front, ir_image, depth_map = get_rgb_and_front_depth()
                        if rgb is not None:
                            extra_frames, extra_paths = get_view_360_capture(frame_idx)
                            if extra_frames is None:
                                time.sleep(0.005)
                                continue

                            rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                            ir_path = os.path.join(RUN_DIR, f"ir_{frame_idx:05d}.png") if ir_image is not None else None
                            depth_path = os.path.join(RUN_DIR, f"depth_{frame_idx:05d}.png") if depth_map is not None else None

                            rgb_source = "cam0" if PRIMARY_RGB_IS_CAM0 else ("realsense" if args.camera == "realsense" else "opencv")
                            depth_source = "realsense_d435i" if args.camera == "realsense" else "opencv"
                            imu_source = depth_source
                            rear_rgb_source = "cam1" if any(
                                os.path.basename(extra_path).startswith("cam1_")
                                for extra_path in extra_paths
                            ) else "none"

                            cam0_path = rgb_path if PRIMARY_RGB_IS_CAM0 else next((p for p in extra_paths if os.path.basename(p).startswith("cam0_")), "")
                            cam1_path = next((p for p in extra_paths if os.path.basename(p).startswith("cam1_")), "")
                            row_data = [
                                time.time(),
                                s_us,
                                t_us,
                                pwm_to_norm(s_us),
                                pwm_to_norm(t_us),
                                depth_front,
                                rgb_path,
                                *build_row_metadata(
                                    rgb_source,
                                    depth_source,
                                    imu_source,
                                    rear_rgb_source,
                                    PREPROCESS_PROFILE,
                                    cam0_path,
                                    cam1_path,
                                    ir_path,
                                    depth_path,
                                ),
                            ]
                            if not write_queue.full():
                                cmd = "FRAME_ALL" if ir_image is not None or depth_map is not None else "FRAME"
                                if cmd == "FRAME_ALL":
                                    write_queue.put((
                                        "FRAME_ALL",
                                        rgb,
                                        ir_image,
                                        depth_map,
                                        row_data,
                                        rgb_path,
                                        PREPROCESS_PROFILE,
                                        ir_path,
                                        depth_path,
                                        extra_frames,
                                    ))
                                else:
                                    write_queue.put((
                                        "FRAME",
                                        rgb,
                                        row_data,
                                        rgb_path,
                                        PREPROCESS_PROFILE,
                                        extra_frames,
                                    ))
                                last_steer_rec = s_us
                                last_throttle_rec = t_us
                                frame_idx += 1
                                last_save_time = now
                                if frame_idx % 2 == 0:
                                    print(f"\rQ:{write_queue.qsize()} | Frame {frame_idx:05d} | S {pwm_to_norm(s_us):+0.3f} | "
                                          f"T {pwm_to_norm(t_us):+0.3f} | D {depth_front:.2f}", end="")
            
            # Small sleep to prevent CPU hogging in this thread
            time.sleep(0.005)
        else:
            # Sleep longer when not recording
            time.sleep(0.1)

# Start the recording thread
threading.Thread(target=recording_worker, daemon=True).start()

# ================= INPUT THREAD =================
recording = False
def input_thread():
    global recording
    while True:
        try:
            key = input().strip().lower()
        except EOFError:
            return
        if key == "":
            recording = not recording
            print(f"\n>>> {'RECORDING' if recording else 'PAUSED'}")
        elif key in ("\x08","\x7f","\b"):
            delete_last_n(cfg.DELETE_N_FRAMES)
        elif key in ("del","d"):
            delete_current_session()

threading.Thread(target=input_thread, daemon=True).start()

# ================= STARTUP INSTRUCTIONS =================
print("\n--- RC CAR DATA COLLECTION ---")
print("Controls:")
print("  ENTER -> Start/Pause recording")
print(f"  BACKSPACE -> Delete last {cfg.DELETE_N_FRAMES} frames")
print("  DEL/d -> Delete current session")
print("  Ctrl+C -> Quit")
print(f"  Save Rate -> {cfg.TARGET_FPS} FPS")
if PRIMARY_RGB_IS_CAM0:
    print(
        f"  Primary RGB -> CAM0 (sensor-id={args.view_360_cam0_sensor_id}, "
        f"save={DEFAULT_OUTPUT_SIZE[0]}x{DEFAULT_OUTPUT_SIZE[1]}, profile={PREPROCESS_PROFILE})"
    )
elif args.primary_rgb_source == "main_camera":
    print("  Primary RGB -> Main camera (RealSense if available)")
if cam_rig is not None and cam_rig.cam1_ok:
    print(
        f"  Rear Preview -> ENABLED (CAM0 sensor-id={args.view_360_cam0_sensor_id}, "
        f"CAM1 sensor-id={args.view_360_cam1_sensor_id}, "
        f"save={args.view_360_save_width}x{args.view_360_save_height}, "
        f"flip={args.view_360_cam0_flip_method}/{args.view_360_cam1_flip_method})"
    )
elif args.view_360 and cam_rig is not None and cam_rig.cam0_ok:
    print("  Rear Preview -> CAM0 available, CAM1 unavailable; continuing without rear sidecar")
elif args.view_360:
    print("  Rear Preview -> REQUESTED but unavailable, continuing without CAM0/CAM1 recording")
print("\n>>> RECORDING will start after pressing ENTER\n")

# ================= UTIL: deadzone =================
def apply_deadzone(value, threshold=cfg.AXIS_DEADZONE):
    return 0.0 if abs(value) < threshold else value

# ================= MAIN LOOP =================
last_steer_sent = -1
last_throttle_sent = -1
last_pca_send_ts = 0.0

# Start camera pipeline immediately to avoid startup lag when recording begins
if CAMERA_ENABLED:
    try:
        realsense_full.start_pipeline()
    except Exception as e:
        print(f"Warning: Camera failed to start: {e}")

try:
    # Ensure neutral at start
    neutralize()
    while True:
        if not USE_NETWORK_CONTROLLER:
            pygame.event.pump()

            # STEERING (left stick X) - invert to match mapping
            raw_steer = -joystick.get_axis(cfg.STEERING_AXIS)
            raw_steer = apply_deadzone(raw_steer)
            
            # Apply Gamma Curve for fine control
            sign = 1 if raw_steer >= 0 else -1
            steer_curved = sign * (abs(raw_steer) ** cfg.STEERING_GAMMA)
            steer = max(min(steer_curved, cfg.STEERING_MAX_SCALE), -cfg.STEERING_MAX_SCALE)

            # THROTTLE selection by mode
            if mode == 3:
                if USE_LEFT_TRIGGER_ONLY:
                    # Use left trigger only for acceleration (0..1)
                    lt = (joystick.get_axis(cfg.LEFT_TRIGGER_AXIS) + 1.0) / 2.0
                    if lt < cfg.AXIS_DEADZONE:
                        lt = 0.0
                    throttle_axis = lt
                else:
                    rt = (joystick.get_axis(cfg.RIGHT_TRIGGER_AXIS) + 1.0) / 2.0
                    lt = (joystick.get_axis(cfg.LEFT_TRIGGER_AXIS)  + 1.0) / 2.0
                    if rt < cfg.AXIS_DEADZONE:
                        rt = 0.0
                    if lt < cfg.AXIS_DEADZONE:
                        lt = 0.0
                    throttle_axis = rt - lt  # -1 -> +1
            else:
                raw_thr = -joystick.get_axis(cfg.THROTTLE_AXIS)
                raw_thr = apply_deadzone(raw_thr)
                throttle_axis = raw_thr

            throttle_axis = max(min(throttle_axis, cfg.THROTTLE_MAX_SCALE), -cfg.THROTTLE_MAX_SCALE)
            steer_us = int(STEERING_CENTER + steer * (STEERING_MAX - STEERING_CENTER))
            throttle_us = int(THROTTLE_CENTER + throttle_axis * (THROTTLE_MAX - THROTTLE_CENTER))
        else:
            # Use network-provided normalized values
            now = time.time()
            if now - net_last_ts > NET_TIMEOUT_S:
                steer_us = STEERING_CENTER
                throttle_us = THROTTLE_CENTER
            else:
                s = max(min(net_steer_norm, cfg.STEERING_MAX_SCALE), -cfg.STEERING_MAX_SCALE)
                t = max(min(net_throttle_norm, cfg.THROTTLE_MAX_SCALE), -cfg.THROTTLE_MAX_SCALE)
                steer_us = int(STEERING_CENTER + s * (STEERING_MAX - STEERING_CENTER))
                throttle_us = int(THROTTLE_CENTER + t * (THROTTLE_MAX - THROTTLE_CENTER))

        # Send to PCA if changed OR at keepalive interval to mirror old constant updates
        send_now = False
        now_ts = time.time()
        if steer_us != last_steer_sent or throttle_us != last_throttle_sent:
            send_now = True
        elif now_ts - last_pca_send_ts >= 0.05:  # 20 Hz keepalive pulses
            send_now = True

        if send_now:
            try:
                pca.set_us(cfg.STEERING_CHANNEL, steer_us)
                pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)
                last_steer_sent = steer_us
                last_throttle_sent = throttle_us
                last_pca_send_ts = now_ts
                
                # Update shared state for the recording thread
                current_steer_us = steer_us
                current_throttle_us = throttle_us
                
            except Exception as e:
                print(f"[PCA ERROR] {e}")

        # Main loop only handles control now. Recording is in background thread.
        time.sleep(0.001) # Reduced sleep to 1ms for higher polling rate

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    print("\nDraining write queue...")
    write_queue.put(None)
    write_queue.join()

    csv_file.close()
    neutralize()
    if not USE_NETWORK_CONTROLLER:
        pygame.quit()
    if cam_rig is not None:
        cam_rig.stop()
    if CAMERA_ENABLED:
        try:
            realsense_full.stop_pipeline()
        except Exception as e:
            print(f"[RealSense stop error] {e}")
    print(f"\nDATA SAVED -> {RUN_DIR}")

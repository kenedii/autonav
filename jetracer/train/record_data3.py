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

# ================= ARGS =================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "opencv", "other"], help="Camera type")
parser.add_argument("--device", type=int, default=0, help="Camera device ID (for opencv)")
parser.add_argument('--record_mode', type=str, default='rgb', choices=['rgb', 'all'], help='Recording mode: "rgb" for control+RGB only, "all" for control+RGB+IR+depth')
parser.add_argument('--control_mode', type=str, default=None, choices=['joystick', 'steer_trigger'], help='Optional override for control mapping: "steer_trigger" uses left trigger for accel')
parser.add_argument('--always_save', action='store_true', help='Save frames at TARGET_FPS even when controls have not changed')
parser.add_argument('--view_360', action='store_true', help='Also record two Jetson CSI cameras connected to CAM0/CAM1')
parser.add_argument('--view_360_cam0_sensor_id', type=int, default=0, help='Jetson CSI sensor-id for the CAM0 360 camera')
parser.add_argument('--view_360_cam1_sensor_id', type=int, default=1, help='Jetson CSI sensor-id for the CAM1 360 camera')
parser.add_argument('--view_360_width', type=int, default=640, help='Capture width for the 360 cameras')
parser.add_argument('--view_360_height', type=int, default=480, help='Capture height for the 360 cameras')
parser.add_argument('--view_360_fps', type=int, default=15, help='Capture FPS for the 360 cameras')
parser.add_argument('--view_360_cam0_flip_method', type=int, default=0, help='nvvidconv flip-method for CAM0 (0-7)')
parser.add_argument('--view_360_cam1_flip_method', type=int, default=0, help='nvvidconv flip-method for CAM1 (0-7)')
args = parser.parse_args()

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
    TARGET_FPS = 2
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
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    @staticmethod
    def build_pipeline(sensor_id, width, height, fps, flip_method):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def start(self):
        if self.cap is not None:
            return True

        pipeline = self.build_pipeline(
            self.sensor_id,
            self.width,
            self.height,
            self.fps,
            self.flip_method,
        )
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"[360] Failed to open {self.label} on sensor-id={self.sensor_id}")
            self.cap.release()
            self.cap = None
            return False

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        print(
            f"[360] {self.label} started on sensor-id={self.sensor_id} "
            f"({self.width}x{self.height}@{self.fps}, flip={self.flip_method})"
        )
        return True

    def _reader(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                break

            ok, frame = self.cap.read()
            if not ok or frame is None:
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
        self.thread = None


class DualCsi360Capture:
    def __init__(self):
        self.cam0 = CsiCameraStream(
            sensor_id=args.view_360_cam0_sensor_id,
            width=args.view_360_width,
            height=args.view_360_height,
            fps=args.view_360_fps,
            flip_method=args.view_360_cam0_flip_method,
            label="CAM0",
        )
        self.cam1 = CsiCameraStream(
            sensor_id=args.view_360_cam1_sensor_id,
            width=args.view_360_width,
            height=args.view_360_height,
            fps=args.view_360_fps,
            flip_method=args.view_360_cam1_flip_method,
            label="CAM1",
        )
        self.enabled = False

    def start(self):
        cam0_ok = self.cam0.start()
        cam1_ok = self.cam1.start()
        self.enabled = cam0_ok and cam1_ok
        if not self.enabled:
            print("[360] Disabling 360 capture because one or more CSI cameras failed to start.")
            self.stop()
        return self.enabled

    def get_frames(self):
        if not self.enabled:
            return None, None
        return self.cam0.get_frame(), self.cam1.get_frame()

    def stop(self):
        self.cam0.stop()
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
view_360_capture = None
VIEW_360_ENABLED = False
view_360_waiting_logged = False

if args.view_360 and CAMERA_ENABLED:
    view_360_capture = DualCsi360Capture()
    VIEW_360_ENABLED = view_360_capture.start()
elif args.view_360:
    print("[360] 360 capture requested but CAMERA_ENABLED=False, so CAM0/CAM1 recording is disabled.")

# ================= MULTI-SESSION SETUP =================
BASE_RUN_DIR = "runs_rgb_depth"
os.makedirs(BASE_RUN_DIR, exist_ok=True)

def create_new_session():
    session_dir = os.path.join(BASE_RUN_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def get_dataset_header():
    if args.record_mode == 'all' and args.camera == 'realsense':
        header = ["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path","ir_path","depth_path"]
    else:
        header = ["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path"]

    if VIEW_360_ENABLED:
        header.extend(["cam0_path", "cam1_path"])

    return header

RUN_DIR = create_new_session()
csv_path = os.path.join(RUN_DIR, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(get_dataset_header())

frame_idx = 0

# ================= WRITER THREAD =================
write_queue = queue.Queue(maxsize=100)

def save_resized_frame(image, path):
    if image is None or path is None:
        return

    image_small = cv2.resize(image, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    cv2.imwrite(path, image_small)

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
            _, rgb, row_data, rgb_path, extra_frames = item
            try:
                save_resized_frame(rgb, rgb_path)
                for extra_path, extra_image in extra_frames:
                    save_resized_frame(extra_image, extra_path)
                writer.writerow(row_data)
                csv_file.flush()
            except Exception as e:
                print(f"Write error: {e}")
        elif cmd == "FRAME_ALL":
            _, rgb, ir_image, depth_map, row_data, rgb_path, ir_path, depth_path, extra_frames = item
            try:
                save_resized_frame(rgb, rgb_path)
                if ir_image is not None and ir_path is not None:
                    cv2.imwrite(ir_path, ir_image)
                if depth_map is not None and depth_path is not None:
                    cv2.imwrite(depth_path, depth_map)
                for extra_path, extra_image in extra_frames:
                    save_resized_frame(extra_image, extra_path)
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
        if args.record_mode == 'all' and args.camera == 'realsense':
            return None, 0.0, None, None
        return None, 0.0

    # If user requested all frames and using RealSense, try to fetch RGB, IR and full depth map
    if args.record_mode == 'all' and args.camera == 'realsense':
        if hasattr(realsense_full, 'get_all_frames'):
            rgb, center_depth, ir_image, depth_map = realsense_full.get_all_frames()
        else:
            rgb, center_depth = realsense_full.get_aligned_frames()
            ir_image, depth_map = None, None
        if rgb is None:
            return None, None, None, None
        return rgb, float(center_depth), ir_image, depth_map

    # Default: RGB + single center depth
    rgb, center_depth = realsense_full.get_aligned_frames()
    if rgb is None:
        return None, None
    return rgb, float(center_depth)

def get_view_360_capture(frame_number):
    global view_360_waiting_logged

    if not VIEW_360_ENABLED or view_360_capture is None:
        return [], []

    cam0_frame, cam1_frame = view_360_capture.get_frames()
    if cam0_frame is None or cam1_frame is None:
        if not view_360_waiting_logged:
            print("\n[360] Waiting for CAM0/CAM1 frames before saving synchronized 360 data...")
            view_360_waiting_logged = True
        return None, None

    view_360_waiting_logged = False
    cam0_path = os.path.join(RUN_DIR, f"cam0_{frame_number:05d}.png")
    cam1_path = os.path.join(RUN_DIR, f"cam1_{frame_number:05d}.png")
    extra_frames = [(cam0_path, cam0_frame), (cam1_path, cam1_frame)]
    extra_paths = [cam0_path, cam1_path]
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
    global frame_idx, RUN_DIR, csv_path, csv_file, writer, recording, view_360_waiting_logged
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
                        # Fetch frame(s) according to requested record_mode
                        if args.record_mode == 'all' and args.camera == 'realsense':
                            rgb, depth_front, ir_image, depth_map = get_rgb_and_front_depth()
                            if rgb is not None:
                                extra_frames, extra_paths = get_view_360_capture(frame_idx)
                                if extra_frames is None:
                                    time.sleep(0.005)
                                    continue
                                rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                                ir_path = os.path.join(RUN_DIR, f"ir_{frame_idx:05d}.png") if ir_image is not None else None
                                depth_path = os.path.join(RUN_DIR, f"depth_{frame_idx:05d}.png") if depth_map is not None else None
                                row_data = [time.time(), s_us, t_us,
                                            pwm_to_norm(s_us), pwm_to_norm(t_us),
                                            depth_front, rgb_path, ir_path, depth_path]
                                row_data.extend(extra_paths)
                                if not write_queue.full():
                                    write_queue.put(("FRAME_ALL", rgb, ir_image, depth_map, row_data, rgb_path, ir_path, depth_path, extra_frames))
                                    last_steer_rec = s_us
                                    last_throttle_rec = t_us
                                    frame_idx += 1
                                    last_save_time = now
                                    if frame_idx % 2 == 0:
                                        print(f"\rQ:{write_queue.qsize()} | Frame {frame_idx:05d} | S {pwm_to_norm(s_us):+0.3f} | "
                                              f"T {pwm_to_norm(t_us):+0.3f} | D {depth_front:.2f}", end="")
                        else:
                            rgb, depth_front = get_rgb_and_front_depth()
                            if rgb is not None:
                                extra_frames, extra_paths = get_view_360_capture(frame_idx)
                                if extra_frames is None:
                                    time.sleep(0.005)
                                    continue
                                rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                                row_data = [time.time(), s_us, t_us,
                                            pwm_to_norm(s_us), pwm_to_norm(t_us),
                                            depth_front, rgb_path]
                                row_data.extend(extra_paths)
                                if not write_queue.full():
                                    write_queue.put(("FRAME", rgb, row_data, rgb_path, extra_frames))
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
if VIEW_360_ENABLED:
    print(
        f"  360 View -> ENABLED (CAM0 sensor-id={args.view_360_cam0_sensor_id}, "
        f"CAM1 sensor-id={args.view_360_cam1_sensor_id})"
    )
elif args.view_360:
    print("  360 View -> REQUESTED but unavailable, continuing without CAM0/CAM1 recording")
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
    if view_360_capture is not None:
        view_360_capture.stop()
    if CAMERA_ENABLED:
        try:
            realsense_full.stop_pipeline()
        except Exception as e:
            print(f"[RealSense stop error] {e}")
    print(f"\nDATA SAVED -> {RUN_DIR}")

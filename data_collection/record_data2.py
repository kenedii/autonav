"""
Minimal RC car data collection with session management:
- Uses only RGB camera and a single front depth value
- Immediate PWM output, low-latency
- Supports multi-session recording and deletion of last N frames or entire session
- Throttle capped at 30% of full speed
"""

import numpy as np
import os
import csv
import time
import threading
from datetime import datetime
import pygame
from smbus2 import SMBus
import cv2
import argparse
import queue
import realsense_full  # Your verified RealSense pipeline

# ================= ARGS =================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "opencv", "other"], help="Camera type")
parser.add_argument("--device", type=int, default=0, help="Camera device ID (for opencv)")
parser.add_argument('--record_mode', type=str, default='rgb', choices=['rgb', 'all'], help='Recording mode: "rgb" for control+RGB only, "all" for control+RGB+IR+depth')
parser.add_argument('--control_mode', type=str, default='joystick', choices=['joystick', 'steer_trigger'], help='Control mapping: "joystick" (default) or "steer_trigger" (steer by stick, left trigger for accel)')
parser.add_argument('--always_save', action='store_true', help='Save frames at TARGET_FPS even when controls have not changed')
args = parser.parse_args()

if args.camera in ["opencv", "other"]:
    realsense_full.set_camera_type("opencv", args.device)
else:
    realsense_full.set_camera_type("realsense")

# ================= CONFIG =================
class Config:
    PCA_ADDR = 0x40
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1
    STEERING_AXIS = 0
    THROTTLE_AXIS = 1
    LEFT_TRIGGER_AXIS = 2
    PWM_FREQ = 50
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    TARGET_FPS = 2
    DELETE_N_FRAMES = 50
    THROTTLE_MAX_SCALE = 0.25  # Max 30% of full speed

cfg = Config()

# ================= PCA9685 =================
class PCA9685:
    def __init__(self, bus=1, address=cfg.PCA_ADDR):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(cfg.PWM_FREQ)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096 / freq_hz - 1
        prescale = int(prescaleval + 0.5)
        self.bus.write_byte_data(self.address, 0x00, 0x10)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, off >> 8)

    def set_us(self, channel, microseconds):
        pulse_length = 1000000 / cfg.PWM_FREQ / 4096
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

pca = PCA9685()
STEERING_CENTER = 1500
THROTTLE_CENTER = 1500
STEERING_MAX = 2000
STEERING_MIN = 1000
THROTTLE_MAX = 2000
THROTTLE_MIN = 1000
pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)

# ================= PYGAME =================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected! Connect Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")

# ================= MULTI-SESSION SETUP =================
BASE_RUN_DIR = "runs_rgb_depth"
os.makedirs(BASE_RUN_DIR, exist_ok=True)

def create_new_session():
    session_dir = os.path.join(BASE_RUN_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

RUN_DIR = create_new_session()
csv_path = os.path.join(RUN_DIR, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
if args.record_mode == 'all' and args.camera == 'realsense':
    writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path","ir_path","depth_path"])
else:
    writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path"])
frame_idx = 0

# ================= WRITER THREAD =================
write_queue = queue.Queue(maxsize=100)

def writer_worker():
    """Background writer: handles FRAME, FRAME_ALL and DELETE commands from the queue."""
    global csv_file, writer
    while True:
        try:
            item = write_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            # Shutdown signal
            write_queue.task_done()
            break

        cmd = item[0]
        if cmd == "FRAME":
            _, rgb, row_data, rgb_path = item
            try:
                cv2.imwrite(rgb_path, rgb)
                writer.writerow(row_data)
                csv_file.flush()
            except Exception as e:
                print(f"Write error: {e}")

        elif cmd == "FRAME_ALL":
            _, rgb, ir_image, depth_map, row_data, rgb_path, ir_path, depth_path = item
            try:
                cv2.imwrite(rgb_path, rgb)
                if ir_image is not None and ir_path is not None:
                    cv2.imwrite(ir_path, ir_image)
                if depth_map is not None and depth_path is not None:
                    cv2.imwrite(depth_path, depth_map)
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
    # Use optimized fetch from realsense_full which works for both RealSense and OpenCV
    if args.record_mode == 'all' and args.camera == 'realsense':
        # Try to get all data (RGB, IR, depth)
        rgb, depth_center, ir_image, depth_map = realsense_full.get_all_frames() if hasattr(realsense_full, 'get_all_frames') else (None, None, None, None)
        if rgb is None:
            return None, None, None, None
        rgb_small = cv2.resize(rgb, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        return rgb_small, depth_center, ir_image, depth_map
    else:
        rgb, depth_center = realsense_full.get_aligned_frames()
        if rgb is None:
            return None, None
        rgb_small = cv2.resize(rgb, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
        return rgb_small, depth_center

def delete_last_n(n):
    global frame_idx
    if frame_idx == 0:
        print("\nNothing to delete")
        return
    
    # Decrement index immediately so new frames overwrite old ones
    frame_idx = max(0, frame_idx - n)
    # Queue the CSV cleanup
    write_queue.put(("DELETE", n))
    print(f"\nRequested delete of last {n} frames -> index reverted to {frame_idx}")

def delete_current_session():
    global frame_idx, csv_file, writer, RUN_DIR, csv_path
    confirm = input(f"\nDelete current session '{RUN_DIR}'? [y/N]: ").strip().lower()
    if confirm == 'y':
        # Clear queue first to avoid writing to deleted session
        with write_queue.mutex:
            write_queue.queue.clear()
            
        csv_file.close()
        for fname in os.listdir(RUN_DIR):
            os.remove(os.path.join(RUN_DIR, fname))
        os.rmdir(RUN_DIR)
        print(f"Session '{RUN_DIR}' deleted!")
        RUN_DIR = create_new_session()
        csv_path = os.path.join(RUN_DIR, "dataset.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        if args.record_mode == 'all' and args.camera == 'realsense':
            writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path","ir_path","depth_path"])
        else:
            writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path"])
        frame_idx = 0

# ================= INPUT THREAD =================
recording = False
def input_thread():
    global recording
    while True:
        key = input().strip().lower()
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
print("  BACKSPACE -> Delete last 50 frames")
print("  DEL/d -> Delete current session")
print("  Ctrl+C -> Quit")
print("\n>>> RECORDING will start after pressing ENTER\n")

# ================= MAIN LOOP =================
last_steer = STEERING_CENTER
last_throttle = THROTTLE_CENTER
MIN_CHANGE_US = 15

try:
    last_save_time = 0
    while True:
        pygame.event.pump()
        # Two control modes: full joystick (default), or steer via stick + left trigger for accel
        if args.control_mode == 'steer_trigger':
            steer = -joystick.get_axis(cfg.STEERING_AXIS)
            lt = (joystick.get_axis(cfg.LEFT_TRIGGER_AXIS) + 1.0) / 2.0
            if lt < 0.03:
                lt = 0.0
            throttle_axis = lt
        else:
            steer = -joystick.get_axis(cfg.STEERING_AXIS)
            throttle_axis = -joystick.get_axis(cfg.THROTTLE_AXIS)

        # Cap throttle to configured max (supports negative for reverse if available)
        throttle_axis = max(min(throttle_axis, cfg.THROTTLE_MAX_SCALE), -cfg.THROTTLE_MAX_SCALE)

        steer_us = int(STEERING_CENTER + steer*(STEERING_MAX - STEERING_CENTER))
        throttle_us = int(THROTTLE_CENTER + throttle_axis*(THROTTLE_MAX - THROTTLE_CENTER))

        pca.set_us(cfg.STEERING_CHANNEL, steer_us)
        pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)

        now = time.time()
        if recording and now - last_save_time >= 1.0 / cfg.TARGET_FPS:
            # Save either when inputs changed beyond threshold OR when user requested always-save
            if args.always_save or abs(steer_us-last_steer) >= MIN_CHANGE_US or abs(throttle_us-last_throttle) >= MIN_CHANGE_US:
                last_steer, last_throttle = steer_us, throttle_us
                if args.record_mode == 'all' and args.camera == 'realsense':
                    rgb, depth_front, ir_image, depth_map = get_rgb_and_front_depth()
                    if rgb is not None:
                        rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                        ir_path = os.path.join(RUN_DIR, f"ir_{frame_idx:05d}.png") if ir_image is not None else None
                        depth_path = os.path.join(RUN_DIR, f"depth_{frame_idx:05d}.png") if depth_map is not None else None
                        if not write_queue.full():
                            row_data = [time.time(), steer_us, throttle_us,
                                        pwm_to_norm(steer_us), pwm_to_norm(throttle_us),
                                        depth_front, rgb_path, ir_path, depth_path]
                            # Save all images
                            write_queue.put(("FRAME_ALL", rgb, ir_image, depth_map, row_data, rgb_path, ir_path, depth_path))
                            frame_idx += 1
                            last_save_time = now
                            print(f"\rQ:{write_queue.qsize()} | Frame {frame_idx:05d} | S {pwm_to_norm(steer_us):+0.3f} | "
                                  f"T {pwm_to_norm(throttle_us):+0.3f} | D {depth_front:.2f}", end="")
                        else:
                            print(f"\r[WARN] Write queue full! Dropping frame.       ", end="")
                else:
                    rgb, depth_front = get_rgb_and_front_depth()
                    if rgb is not None:
                        rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                        if not write_queue.full():
                            row_data = [time.time(), steer_us, throttle_us,
                                        pwm_to_norm(steer_us), pwm_to_norm(throttle_us),
                                        depth_front, rgb_path]
                            write_queue.put(("FRAME", rgb, row_data, rgb_path))
                            frame_idx += 1
                            last_save_time = now
                            print(f"\rQ:{write_queue.qsize()} | Frame {frame_idx:05d} | S {pwm_to_norm(steer_us):+0.3f} | "
                                  f"T {pwm_to_norm(throttle_us):+0.3f} | D {depth_front:.2f}", end="")
                        else:
                            print(f"\r[WARN] Write queue full! Dropping frame.       ", end="")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    csv_file.close()
    pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
    pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)
    pygame.quit()
    if realsense_full.pipeline:
        realsense_full.pipeline.stop()
        print("[RealSense] Pipeline stopped.")
    print(f"\nDATA SAVED -> {RUN_DIR}")


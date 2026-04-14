# record_data.py
# Python Script to record training data to the Jetson Nano's MicroSD card.
# After xbox controller config is complete and configuration is complete in this script,
# This script allows the user to drive the car around the track using the XBOX controller.
# It will save the flattened RGB, IR, and depth data as independent variables, and the steering and acceleration values are the target variables of the dataset.
# This data can be used to predict steering and acceleration values when given an image of what the car sees.
import pyrealsense2 as rs
import numpy as np
import pygame
import time
import os
import csv
import threading
import argparse
import cv2
from datetime import datetime
import Jetson.GPIO as GPIO

# ================= ARGS =================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "opencv", "other"], help="Camera type: realsense or opencv")
parser.add_argument("--device", type=int, default=0, help="Camera device ID (for opencv)")
args = parser.parse_args()

# ================= CONFIG =================
STEERING_PIN = 0  # Not used, car controlled via PCA9685 directly
THROTTLE_PIN = 1  # Not used
TARGET_FPS = 8
RUN_DIR = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DELETE_N_FRAMES = 50

os.makedirs(RUN_DIR, exist_ok=True)
IMG_DIR = os.path.join(RUN_DIR, "frames")
os.makedirs(IMG_DIR, exist_ok=True)

CSV_PATH = os.path.join(RUN_DIR, "labels.csv")
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["filename", "steer", "throttle"])

# ================== CAMERA SETUP ==================
latest_frames = {"rgb": None, "ir": None, "depth": None}
lock = threading.Lock()

pipeline = None
cap = None

if args.camera == "realsense":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, TARGET_FPS)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, TARGET_FPS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, TARGET_FPS)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
else:
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"OpenCV Camera started on device {args.device}")

def camera_thread():
    global latest_frames
    if args.camera == "realsense":
        while True:
            try:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                ir = aligned.get_infrared_frame(1)
                if color and depth and ir:
                    with lock:
                        latest_frames["rgb"] = np.asanyarray(color.get_data())
                        latest_frames["depth"] = np.asanyarray(depth.get_data())
                        latest_frames["ir"] = np.asanyarray(ir.get_data())
            except Exception as e:
                print(f"RS Error: {e}")
    else:
        while True:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Create dummy depth/ir matching size
                    h, w, c = rgb.shape
                    dummy_depth = np.zeros((h, w), dtype=np.uint16)
                    dummy_ir = np.zeros((h, w), dtype=np.uint8)
                    
                    with lock:
                        latest_frames["rgb"] = rgb
                        latest_frames["depth"] = dummy_depth
                        latest_frames["ir"] = dummy_ir
                else:
                    time.sleep(0.01)

threading.Thread(target=camera_thread, daemon=True).start()

# ================== JOYSTICK ==================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected!")
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Map left stick for steering (X axis) and throttle (Y axis)
# Reverse values as needed
def get_controller_input():
    pygame.event.pump()
    steer = joystick.get_axis(0)  # Left stick horizontal
    throttle = -joystick.get_axis(1)  # Left stick vertical (up positive)
    return steer, throttle

# ================== PCA9685 / Car Control ==================
# Use your existing PCA9685 class or methods here to send PWM signals
# For demonstration, we just print values

# ================== MAIN LOOP ==================
recording = False
frame_idx = 0
last_time = 0

def delete_last_n(n):
    global frame_idx
    n = min(n, frame_idx)
    if n <= 0:
        print("Nothing to delete")
        return
    with open(CSV_PATH, "r") as f:
        lines = f.readlines()
    with open(CSV_PATH, "w") as f:
        f.writelines(lines[:-n])
    for i in range(frame_idx - n, frame_idx):
        path = os.path.join(IMG_DIR, f"{i:06d}.npz")
        if os.path.exists(path):
            os.remove(path)
    frame_idx -= n
    print(f"Deleted last {n} frames, now at {frame_idx}")

def input_thread():
    global recording
    while True:
        key = input()
        if key == "":
            recording = not recording
            print(f"{'RECORDING' if recording else 'PAUSED'}")
        elif key in ("\x08", "\x7f", "\b"):
            delete_last_n(DELETE_N_FRAMES)

threading.Thread(target=input_thread, daemon=True).start()

print("\n--- LOW-LATENCY DATA COLLECTION ---")
print("Controls:")
print("   ENTER      -> Start / Pause")
print("   BACKSPACE  -> Delete last 50 frames")
print("Ctrl+C to quit\n")

try:
    while True:
        now = time.time()
        if not recording or now - last_time < 1.0 / TARGET_FPS:
            time.sleep(0.01)
            continue

        # ================= CONTROLLER =================
        steer, throttle = get_controller_input()

        # TODO: Replace this with actual PCA9685 car control
        print(f"Steer: {steer:+.2f} | Throttle: {throttle:+.2f}", end="\r")

        # ================= CAMERA =================
        with lock:
            rgb = latest_frames["rgb"]
            depth = latest_frames["depth"]
            ir = latest_frames["ir"]
        if rgb is None or depth is None or ir is None:
            continue

        # ================= SAVE =================
        fname = f"{frame_idx:06d}.npz"
        np.savez_compressed(os.path.join(IMG_DIR, fname),
                            rgb=rgb, depth=depth, ir=ir)
        writer.writerow([fname, steer, throttle])
        csv_file.flush()

        frame_idx += 1
        last_time = now

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    if args.camera == "realsense" and pipeline:
        pipeline.stop()
    elif cap:
        cap.release()
        
    csv_file.close()
    pygame.quit()
    print(f"DATA SAVED -> {RUN_DIR}")

    # Write dataset info
    with open(os.path.join(RUN_DIR, "README.txt"), "w") as f:
        f.write("""LOW-LATENCY DATASET
Features: rgb (640x480x3), depth (640x480), ir (640x480)
Target: steering [-1,1], throttle [-1,1]
Instructions:
- Use the Xbox controller left stick for steering and throttle.
- Press ENTER to start/pause recording.
- Press BACKSPACE to delete last 50 frames.
- Process raw .npz files on a more powerful computer for training.
""")

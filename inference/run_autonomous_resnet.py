# run_autonomous_resnet.py
# Fully working autonomous drive script to run the resnet model for Jetson Nano + RealSense + LaTrax
# Uses TensorRT and sends control commands over serial to a Raspberry Pi Pico.
# Will automatically default to using PyTorch to run the model if TensorRT can't be used.
# Adjust MODEL_TRT_PATH or MODEL_PYTORCH_PATH if needed.
#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import time
import signal
import argparse
import atexit

try:
    import serial
except ImportError:
    serial = None

import realsense_full


# ================= ARGUMENTS =================

parser = argparse.ArgumentParser()

parser.add_argument("--backend", default="cuda")
parser.add_argument("--exp", type=int, default=3)
parser.add_argument("--arch", default="resnet34")

parser.add_argument(
    "--camera",
    default="realsense",
    choices=["realsense","opencv"]
)

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--throttle", default="0.3")
parser.add_argument("--serial-port", default="/dev/ttyACM0")
parser.add_argument("--serial-baud", type=int, default=115200)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH = 160
IMG_HEIGHT = 120


# ================= PICO SERIAL CONTROL =================

STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1

STEERING_CENTER = 1500
THROTTLE_CENTER = 1500

STEERING_RANGE = 500
THROTTLE_RANGE = 500


class PicoSerialController:

    def __init__(self, port, baudrate):
        if serial is None:
            raise RuntimeError("pyserial is required for Pico serial control")

        self.ser = serial.Serial(
            port,
            baudrate,
            timeout=0.2,
            write_timeout=0.2,
        )
        # Let USB CDC settle before the first command.
        time.sleep(1.0)

    def set_us(self, channel, us):
        us = int(max(500, min(2500, us)))
        self.ser.write(f"SET {int(channel)} {us}\n".encode("ascii"))
        self.ser.flush()

    def close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass


motor_controller = PicoSerialController(args.serial_port, args.serial_baud)


# ================= MODEL =================

EXPERIMENT_FEATURES = {
    1: ['rgb_path','cam1_path','ir_path','depth_path'],
    2: ['rgb_path','ir_path','depth_path'],
    3: ['rgb_path'],
}

FEATURES = EXPERIMENT_FEATURES.get(args.exp,['rgb_path'])

IN_CHANNELS = sum(1 if ('depth' in f or 'ir' in f) else 3 for f in FEATURES)


def build_model():

    model = getattr(models,args.arch)(weights=None)

    out_features = 512 if args.arch in ['resnet18','resnet34'] else 2048

    if IN_CHANNELS != 3:

        model.conv1 = nn.Conv2d(
            IN_CHANNELS,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

    base = nn.Sequential(*list(model.children())[:-2])

    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(out_features,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128,2),
        nn.Tanh()
    )

    return nn.Sequential(base,head)


def load_model():

    if os.path.exists("best_model_trt.pth"):

        from torch2trt import TRTModule

        print("[LOAD] TensorRT engine")

        model = TRTModule()
        model.load_state_dict(torch.load("best_model_trt.pth"))

        return model

    print("[LOAD] PyTorch model")

    model = build_model()
    model.load_state_dict(torch.load("best_model.pth",map_location=DEVICE))

    return model.to(DEVICE).eval()


model = load_model()


# ================= CAMERA =================

print("Starting camera pipeline...")

if args.camera == "opencv":
    realsense_full.set_camera_type("opencv", args.device)
else:
    realsense_full.set_camera_type("realsense")

realsense_full.start_pipeline()


# ================= PREPROCESS =================

def preprocess(frame):

    img = cv2.resize(frame,(IMG_WIDTH,IMG_HEIGHT))

    # Already RGB from realsense_full
    img = img.astype(np.float32)/255.0

    img = np.transpose(img,(2,0,1))

    if IN_CHANNELS > 3:

        reps = int(np.ceil(IN_CHANNELS/3))
        img = np.tile(img,(reps,1,1))[:IN_CHANNELS]

    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    return tensor


# ================= SAFETY =================

def cleanup():

    try:
        motor_controller.set_us(STEERING_CHANNEL,STEERING_CENTER)
        motor_controller.set_us(THROTTLE_CHANNEL,THROTTLE_CENTER)
        motor_controller.close()
    except:
        pass

    try:
        realsense_full.stop_pipeline()
    except:
        pass

    print("Safety Neutralization")


atexit.register(cleanup)


def stop(sig,frame):
    cleanup()
    exit(0)


signal.signal(signal.SIGINT,stop)


# ================= ESC ARM =================

print("Arming ESC...")

for _ in range(20):

    motor_controller.set_us(STEERING_CHANNEL,STEERING_CENTER)
    motor_controller.set_us(THROTTLE_CHANNEL,THROTTLE_CENTER)

    time.sleep(0.05)


# ================= MAIN LOOP =================

print("Autonomous Driving Started")

last=time.time()
count=0

while True:

    rgb, depth, ir, depth_map = realsense_full.get_all_frames()

    if rgb is None:
        continue

    tensor = preprocess(rgb)

    with torch.no_grad():

        out = model(tensor)

        steer = float(out[0][0])
        throttle_model = float(out[0][1])

    steer = np.clip(steer,-1,1)

    if args.throttle == "model":
        throttle = np.clip(throttle_model,-1,1)
    else:
        throttle = float(args.throttle)

    steer_us = int(STEERING_CENTER + steer*STEERING_RANGE)
    throttle_us = int(THROTTLE_CENTER + throttle*THROTTLE_RANGE)

    motor_controller.set_us(STEERING_CHANNEL,steer_us)
    motor_controller.set_us(THROTTLE_CHANNEL,throttle_us)

    count+=1
    now=time.time()

    if now-last>2:

        fps=count/(now-last)

        print(
            f"FPS:{fps:.1f} | "
            f"Steer:{steer:+.2f} ({steer_us}us) | "
            f"Throttle:{throttle:+.2f} ({throttle_us}us)"
        )

        count=0
        last=now

# run_autonomous_resnet.py
# Fully working autonomous drive script to run the resnet model for Jetson Nano + RealSense + LaTrax
# Uses TensorRT and sends control commands over serial to a Raspberry Pi Pico.
# Will automatically default to using PyTorch to run the model if TensorRT can't be used.
# Adjust MODEL_TRT_PATH or MODEL_PYTORCH_PATH if needed.
#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import time
import signal
import argparse
import atexit
import threading

try:
    import serial
except ImportError:
    serial = None

try:
    from smbus2 import SMBus
except ImportError:
    SMBus = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_COLLECTION_DIR = os.path.join(REPO_ROOT, "data_collection")
if DATA_COLLECTION_DIR not in sys.path:
    sys.path.insert(0, DATA_COLLECTION_DIR)


# ================= ARGUMENTS =================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--backend",
    default="cuda",
    choices=["cuda", "rknn"],
)
parser.add_argument("--exp", type=int, default=3)
parser.add_argument("--arch", default="resnet34")
parser.add_argument("--model-path", default="best_model.pth")
parser.add_argument("--trt-model-path", default="best_model_trt.pth")
parser.add_argument("--rknn-model-path", default="best_model.rknn")

parser.add_argument(
    "--camera",
    default="realsense",
    choices=["realsense", "opencv", "cam0"]
)

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--cam-sensor-id", type=int, default=0)
parser.add_argument("--cam-width", type=int, default=640)
parser.add_argument("--cam-height", type=int, default=480)
parser.add_argument("--cam-fps", type=int, default=15)
parser.add_argument("--cam-flip-method", type=int, default=2)
parser.add_argument(
    "--cam-backend",
    default="auto",
    choices=["auto", "argus", "v4l2"],
)
parser.add_argument(
    "--controller-backend",
    default="pca9685",
    choices=["pca9685", "pico"]
)
parser.add_argument("--throttle", default="0.3")
parser.add_argument("--serial-port", default="/dev/ttyACM0")
parser.add_argument("--serial-baud", type=int, default=115200)
parser.add_argument("--pca-bus", type=int, default=1)
parser.add_argument("--pca-address", type=lambda value: int(value, 0), default=0x40)
parser.add_argument("--throttle-scale", type=float, default=None)
parser.set_defaults(invert_steering=None)
steering_group = parser.add_mutually_exclusive_group()
steering_group.add_argument("--invert-steering", dest="invert_steering", action="store_true")
steering_group.add_argument("--no-invert-steering", dest="invert_steering", action="store_false")
parser.add_argument("--preprocess-profile", default=None)
parser.add_argument("--debug-timings", action="store_true")
parser.add_argument(
    "--realsense-stream-mode",
    default="auto",
    choices=["auto", "rgb_only", "rgb_depth_ir"],
)
parser.add_argument("--warmup-iters", type=int, default=10)

args = parser.parse_args()

import realsense_full
from preprocess_utils import apply_preprocess_profile, infer_preprocess_profile

USE_RKNN = args.backend == "rknn"
DEVICE = torch.device("cuda" if (not USE_RKNN and torch.cuda.is_available()) else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

IMG_WIDTH = 160
IMG_HEIGHT = 120


# ================= PICO SERIAL CONTROL =================

STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1

STEERING_CENTER = 1500
THROTTLE_CENTER = 1500

STEERING_RANGE = 500
THROTTLE_RANGE = 500
STEERING_MIN = STEERING_CENTER - STEERING_RANGE
STEERING_MAX = STEERING_CENTER + STEERING_RANGE
THROTTLE_MIN = THROTTLE_CENTER - THROTTLE_RANGE
THROTTLE_MAX = THROTTLE_CENTER + THROTTLE_RANGE


def should_invert_steering():
    if args.invert_steering is not None:
        return args.invert_steering
    return "autonav-v" in args.model_path.lower()


def throttle_output_scale():
    if args.throttle_scale is not None:
        return args.throttle_scale
    if "autonav-v2" in args.model_path.lower():
        return 3.33
    return 1.0


def infer_runtime_preprocess_profile():
    if args.camera == "cam0":
        camera_configs = [{"role": "primary_rgb", "type": "csi", "sensor_id": args.cam_sensor_id}]
    elif args.camera == "opencv":
        camera_configs = [{"role": "primary_rgb", "type": "opencv"}]
    else:
        camera_configs = [{"role": "primary_rgb", "type": "realsense"}]

    return infer_preprocess_profile(
        camera_configs=camera_configs,
        explicit_profile=args.preprocess_profile,
    )


def torchvision_kwargs():
    kwargs = {}
    try:
        from packaging.version import Version
        import torchvision
    except ImportError:
        Version = None
        torchvision = None

    if Version is not None and torchvision is not None and Version(torchvision.__version__) >= Version("0.13.0"):
        kwargs["weights"] = None
    else:
        kwargs["pretrained"] = False
    return kwargs


class RKNNRuntime:

    def __init__(self, model_path):
        self.model_path = model_path
        self.rknn = None
        self._load()

    def _load(self):
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            from rknn.api import RKNN as RKNNLite

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model: %s" % self.model_path)

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

    def infer(self, tensor_cpu):
        # RKNN runtime here follows the NCHW float32 contract used by fleet client.
        if hasattr(tensor_cpu, "detach"):
            inp = tensor_cpu.detach().cpu().numpy().astype(np.float32)
        else:
            inp = np.asarray(tensor_cpu, dtype=np.float32)
        outputs = self.rknn.inference(inputs=[inp])
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def close(self):
        try:
            if self.rknn is not None:
                self.rknn.release()
        except Exception:
            pass


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


class PCA9685Controller:

    def __init__(self, bus_num, address):
        if SMBus is None:
            raise RuntimeError("smbus2 is required for PCA9685 control")

        self.bus = SMBus(bus_num)
        self.address = address
        self.last_pwm = {}
        self.desired_pwm = {}
        self.lock = threading.Lock()
        self.update_event = threading.Event()
        self.stop_event = threading.Event()
        self.set_pwm_freq(50)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096.0 / freq_hz - 1.0
        prescale = int(prescaleval + 0.5)
        self.bus.write_byte_data(self.address, 0x00, 0x10)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        # Keep auto-increment enabled so block writes update the full channel register set.
        self.bus.write_byte_data(self.address, 0x00, 0xA1)

    def _write_pwm(self, channel, on, off):
        key = (int(on), int(off))
        if self.last_pwm.get(channel) == key:
            return

        base_reg = 0x06 + 4 * channel
        self.bus.write_i2c_block_data(
            self.address,
            base_reg,
            [
                on & 0xFF,
                (on >> 8) & 0xFF,
                off & 0xFF,
                (off >> 8) & 0xFF,
            ],
        )
        self.last_pwm[channel] = key

    def _flush_pending(self):
        with self.lock:
            pending = dict(self.desired_pwm)

        for channel, key in pending.items():
            self._write_pwm(channel, key[0], key[1])

    def _writer_loop(self):
        while not self.stop_event.is_set():
            self.update_event.wait(0.02)
            self.update_event.clear()
            self._flush_pending()

    def set_pwm(self, channel, on, off):
        key = (int(on), int(off))
        with self.lock:
            self.desired_pwm[channel] = key
        self.update_event.set()

    def set_us(self, channel, microseconds):
        pulse_length = 1000000.0 / 50.0 / 4096.0
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

    def close(self):
        try:
            self._flush_pending()
            self.stop_event.set()
            self.update_event.set()
            if self.writer_thread.is_alive():
                self.writer_thread.join(timeout=0.5)
            self.bus.close()
        except Exception:
            pass


def build_motor_controller():
    if args.controller_backend == "pico":
        print("[CTRL] Using Pico serial controller on %s" % args.serial_port)
        return PicoSerialController(args.serial_port, args.serial_baud)

    print("[CTRL] Using PCA9685 on I2C bus %s addr %s" % (args.pca_bus, hex(args.pca_address)))
    return PCA9685Controller(args.pca_bus, args.pca_address)


motor_controller = build_motor_controller()
PREPROCESS_PROFILE = infer_runtime_preprocess_profile()
print("[PRE] Using preprocess profile: %s" % PREPROCESS_PROFILE)


# ================= MODEL =================

EXPERIMENT_FEATURES = {
    1: ['rgb_path','cam1_path','ir_path','depth_path'],
    2: ['rgb_path','ir_path','depth_path'],
    3: ['rgb_path'],
}

FEATURES = EXPERIMENT_FEATURES.get(args.exp,['rgb_path'])

IN_CHANNELS = sum(1 if ('depth' in f or 'ir' in f) else 3 for f in FEATURES)


def resolve_realsense_stream_config():
    if args.realsense_stream_mode == "rgb_only":
        return {
            "enable_depth": False,
            "enable_ir": False,
            "enable_depth_preview": False,
            "label": "rgb_only",
        }

    if args.realsense_stream_mode == "rgb_depth_ir":
        return {
            "enable_depth": True,
            "enable_ir": True,
            "enable_depth_preview": False,
            "label": "rgb_depth_ir",
        }

    if FEATURES == ["rgb_path"]:
        return {
            "enable_depth": False,
            "enable_ir": False,
            "enable_depth_preview": False,
            "label": "auto->rgb_only",
        }

    return {
        "enable_depth": True,
        "enable_ir": True,
        "enable_depth_preview": False,
        "label": "auto->rgb_depth_ir",
    }


def build_model():

    model = getattr(models,args.arch)(**torchvision_kwargs())

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

    if USE_RKNN:
        if not os.path.exists(args.rknn_model_path):
            raise FileNotFoundError("RKNN model not found: %s" % args.rknn_model_path)
        print("[LOAD] RKNN model: %s" % args.rknn_model_path)
        return RKNNRuntime(args.rknn_model_path)

    if os.path.exists(args.trt_model_path):

        from torch2trt import TRTModule

        print("[LOAD] TensorRT engine: %s" % args.trt_model_path)

        model = TRTModule()
        model.load_state_dict(torch.load(args.trt_model_path))
        return model.eval()

    print("[LOAD] PyTorch model: %s" % args.model_path)

    model = build_model()
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(checkpoint)

    return model.to(DEVICE).eval()


model = load_model()


def warmup_model():
    if USE_RKNN:
        print("[WARMUP] RKNN backend selected; skipping CUDA warmup")
        return

    if args.warmup_iters <= 0:
        return

    print("[WARMUP] Running %d warmup inference passes..." % args.warmup_iters)
    dummy = torch.zeros((1, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH), device=DEVICE)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.warmup_iters):
            model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print("[WARMUP] Ready (%.1f ms total)" % elapsed_ms)


warmup_model()


# ================= CAMERA =================

print("Starting camera pipeline...")

if args.camera == "opencv":
    realsense_full.set_camera_type("opencv", args.device)
elif args.camera == "cam0":
    realsense_full.set_camera_type(
        "csi",
        sensor_id=args.cam_sensor_id,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.cam_fps,
        flip_method=args.cam_flip_method,
        backend=args.cam_backend,
    )
else:
    realsense_stream_config = resolve_realsense_stream_config()
    print("[RealSense] Runtime stream mode: %s" % realsense_stream_config["label"])
    realsense_full.set_camera_type(
        "realsense",
        enable_depth=realsense_stream_config["enable_depth"],
        enable_ir=realsense_stream_config["enable_ir"],
        enable_depth_preview=realsense_stream_config["enable_depth_preview"],
    )

realsense_full.start_pipeline()


# ================= PREPROCESS =================

def preprocess_to_cpu_tensor(frame):
    img = apply_preprocess_profile(frame, PREPROCESS_PROFILE)
    if img is None:
        raise RuntimeError("Camera frame preprocessing returned no image")

    # Already RGB from realsense_full. Build a contiguous CHW float tensor once.
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)
    img *= 1.0 / 255.0

    if IN_CHANNELS > 3:
        reps = int(np.ceil(IN_CHANNELS / 3))
        img = np.tile(img, (reps, 1, 1))[:IN_CHANNELS]

    tensor = torch.from_numpy(img).unsqueeze(0)
    if DEVICE.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor


class AsyncFramePreprocessor:

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready_tensor = None
        self.ready_seq = 0
        self.last_source_seq = 0
        self.preprocess_total = 0.0
        self.preprocess_count = 0
        self.error = None
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                frame, seq = realsense_full.get_rgb_frame_and_seq(copy_frame=False)
                if frame is None or seq == self.last_source_seq:
                    time.sleep(0.001)
                    continue

                preprocess_start = time.perf_counter()
                tensor = preprocess_to_cpu_tensor(frame)
                preprocess_elapsed = time.perf_counter() - preprocess_start

                with self.lock:
                    self.ready_tensor = tensor
                    self.ready_seq = seq
                    self.last_source_seq = seq
                    self.preprocess_total += preprocess_elapsed
                    self.preprocess_count += 1
            except Exception as exc:
                with self.lock:
                    self.error = exc
                return

    def get_latest(self, last_consumed_seq):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Background preprocessing failed") from self.error
            if self.ready_tensor is None or self.ready_seq == last_consumed_seq:
                return None, last_consumed_seq
            return self.ready_tensor, self.ready_seq

    def consume_stats(self):
        with self.lock:
            preprocess_total = self.preprocess_total
            preprocess_count = self.preprocess_count
            self.preprocess_total = 0.0
            self.preprocess_count = 0
        return preprocess_total, preprocess_count

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)


frame_preprocessor = AsyncFramePreprocessor()


def throttle_to_us(throttle):

    throttle = float(np.clip(throttle, -1.0, 1.0))

    if throttle >= 0.0:
        span = THROTTLE_MAX - THROTTLE_CENTER
    else:
        span = THROTTLE_CENTER - THROTTLE_MIN

    return int(THROTTLE_CENTER + throttle * span)


# ================= SAFETY =================

def cleanup():

    try:
        frame_preprocessor.stop()
    except:
        pass

    try:
        if USE_RKNN and hasattr(model, "close"):
            model.close()
    except:
        pass

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
print("[RUNTIME] backend=%s controller=%s" % (args.backend, args.controller_backend))

last=time.time()
count=0
capture_total = 0.0
preprocess_total = 0.0
inference_total = 0.0
control_total = 0.0
last_frame_seq = 0

with torch.inference_mode():
    while True:
        capture_start = time.perf_counter()
        tensor_cpu, frame_seq = frame_preprocessor.get_latest(last_frame_seq)
        capture_total += time.perf_counter() - capture_start

        if tensor_cpu is None:
            time.sleep(0.001)
            continue

        last_frame_seq = frame_seq

        inference_start = time.perf_counter()
        if USE_RKNN:
            output = model.infer(tensor_cpu)
        else:
            preprocess_start = time.perf_counter()
            tensor = tensor_cpu.to(DEVICE, non_blocking=True)
            preprocess_total += time.perf_counter() - preprocess_start
            out = model(tensor)
            output = out.detach().float().cpu().numpy()[0]
        steer = float(output[0])
        throttle_model = float(output[1]) if len(output) > 1 else 0.0
        inference_total += time.perf_counter() - inference_start

        if should_invert_steering():
            steer *= -1.0

        steer = np.clip(steer,-1,1)

        if args.throttle == "model":
            throttle = np.clip(throttle_model * throttle_output_scale(), -1, 1)
        else:
            throttle = float(args.throttle)

        steer_us = int(STEERING_CENTER + steer*STEERING_RANGE)
        throttle_us = throttle_to_us(throttle)

        control_start = time.perf_counter()
        motor_controller.set_us(STEERING_CHANNEL,steer_us)
        motor_controller.set_us(THROTTLE_CHANNEL,throttle_us)
        control_total += time.perf_counter() - control_start

        count+=1
        now=time.time()

        if now-last>2:

            fps=count/(now-last)
            thread_preprocess_total, thread_preprocess_count = frame_preprocessor.consume_stats()
            avg_thread_preprocess_ms = 0.0
            if thread_preprocess_count > 0:
                avg_thread_preprocess_ms = thread_preprocess_total * 1000.0 / thread_preprocess_count

            print(
                f"FPS:{fps:.1f} | "
                f"Steer:{steer:+.2f} ({steer_us}us) | "
                f"Throttle:{throttle:+.2f} ({throttle_us}us)"
            )
            if args.debug_timings and count > 0:
                print(
                    "Timings(ms): "
                    f"fetch={capture_total * 1000.0 / count:.1f} "
                    f"host_to_device={preprocess_total * 1000.0 / count:.1f} "
                    f"preprocess_thread={avg_thread_preprocess_ms:.1f} "
                    f"infer={inference_total * 1000.0 / count:.1f} "
                    f"control={control_total * 1000.0 / count:.1f}"
                )

            count=0
            last=now
            capture_total = 0.0
            preprocess_total = 0.0
            inference_total = 0.0
            control_total = 0.0

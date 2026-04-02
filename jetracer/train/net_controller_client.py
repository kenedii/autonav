#!/usr/bin/env python3
"""
Network Xbox controller sender.
Run this on your laptop with the Xbox controller plugged in.
It reads the controller via pygame and sends normalized axes over UDP to the Jetson.

Packet format (JSON over UDP): {"s": steer_norm, "t": throttle_norm, "ts": epoch_seconds}
- steer_norm: -1.0 (left) .. +1.0 (right)
- throttle_norm: -1.0 (reverse) .. +1.0 (forward)
"""
import os
import time
import json
import socket
import pygame

# ===== CONFIGURE THESE =====
JETSON_HOST = os.environ.get("JETSON_HOST", "192.168.1.50")  # set to your Jetson IP
JETSON_PORT = int(os.environ.get("JETSON_PORT", "5007"))
# Hotspot-friendly: slightly lower rate to reduce bursts
SEND_HZ = 40.0
AXIS_DEADZONE = 0.03
STEERING_AXIS = 0          # left stick X
THROTTLE_AXIS = 1          # left stick Y (invert below)
USE_TRIGGERS_MODE3 = os.environ.get("USE_TRIGGERS_MODE3", "False").lower() in ("1", "true", "yes")
RIGHT_TRIGGER_AXIS = int(os.environ.get("RIGHT_TRIGGER_AXIS", "5"))
LEFT_TRIGGER_AXIS = int(os.environ.get("LEFT_TRIGGER_AXIS", "2"))
DEBUG_TRIGGERS = os.environ.get("DEBUG_TRIGGERS", "0").lower() in ("1", "true", "yes")
DEBUG_AXES = os.environ.get("DEBUG_AXES", "0").lower() in ("1", "true", "yes")
TRIGGER_CALIBRATION_SEC = float(os.environ.get("TRIGGER_CALIBRATION_SEC", "1.0"))


def apply_deadzone(v, dz=AXIS_DEADZONE):
    return 0.0 if abs(v) < dz else v


def clamp01(v):
    return max(0.0, min(1.0, v))


def calibrate_trigger_rest(js, duration_s=1.0):
    samples = max(1, int(duration_s * 100))
    rt_values = []
    lt_values = []
    print(
        "Calibrating trigger rest positions. Leave RT/LT untouched "
        f"for {duration_s:.1f}s..."
    )
    for _ in range(samples):
        pygame.event.pump()
        rt_values.append(float(js.get_axis(RIGHT_TRIGGER_AXIS)))
        lt_values.append(float(js.get_axis(LEFT_TRIGGER_AXIS)))
        time.sleep(duration_s / samples)
    rt_rest = sum(rt_values) / len(rt_values)
    lt_rest = sum(lt_values) / len(lt_values)
    print(
        f"Trigger neutral calibration complete: "
        f"RT axis {RIGHT_TRIGGER_AXIS} rest={rt_rest:+.3f}, "
        f"LT axis {LEFT_TRIGGER_AXIS} rest={lt_rest:+.3f}"
    )
    return rt_rest, lt_rest


def trigger_magnitude(raw_value, rest_value, dz=AXIS_DEADZONE):
    delta = abs(float(raw_value) - float(rest_value))
    if delta <= dz:
        return 0.0
    max_delta = max(abs(1.0 - rest_value), abs(-1.0 - rest_value), 1e-6)
    scaled = (delta - dz) / max(max_delta - dz, 1e-6)
    return clamp01(scaled)


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No joystick detected. Plug in controller.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Joystick: {js.get_name()}")
    print(f"Sending to udp://{JETSON_HOST}:{JETSON_PORT} at ~{SEND_HZ} Hz")
    print(f"Joystick axes detected: {js.get_numaxes()}")

    rt_rest = -1.0
    lt_rest = -1.0
    if USE_TRIGGERS_MODE3:
        print(
            "Trigger mode enabled: left stick steers, RT accelerates, LT brakes/reverses."
        )
        rt_rest, lt_rest = calibrate_trigger_rest(js, duration_s=TRIGGER_CALIBRATION_SEC)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / SEND_HZ
    next_send = time.time()

    counter = 0
    next_debug = time.time()
    next_axis_debug = time.time()
    try:
        while True:
            pygame.event.pump()
            # Steering
            raw_steer = -js.get_axis(STEERING_AXIS)  # invert to match car mapping
            steer = apply_deadzone(raw_steer)
            # Throttle: default to left stick Y; triggers optional via USE_TRIGGERS_MODE3
            if USE_TRIGGERS_MODE3:
                raw_rt = js.get_axis(RIGHT_TRIGGER_AXIS)
                raw_lt = js.get_axis(LEFT_TRIGGER_AXIS)
                rt = trigger_magnitude(raw_rt, rt_rest)
                lt = trigger_magnitude(raw_lt, lt_rest)
                throttle = rt - lt  # -1..1
                if DEBUG_TRIGGERS and time.time() >= next_debug:
                    print(
                        f"[TRIG] rt_raw={raw_rt:+.3f} lt_raw={raw_lt:+.3f} "
                        f"rt_rest={rt_rest:+.3f} lt_rest={lt_rest:+.3f} "
                        f"rt={rt:+.2f} lt={lt:+.2f} throttle={throttle:+.2f}"
                    )
                    next_debug = time.time() + 0.5
            else:
                raw_thr = -js.get_axis(THROTTLE_AXIS)
                throttle = apply_deadzone(raw_thr)

            # Clamp
            steer = max(min(steer, 1.0), -1.0)
            throttle = max(min(throttle, 1.0), -1.0)

            if DEBUG_AXES and time.time() >= next_axis_debug:
                axis_values = [
                    f"a{axis_index}={js.get_axis(axis_index):+.3f}"
                    for axis_index in range(js.get_numaxes())
                ]
                print("[AXES] " + " ".join(axis_values))
                next_axis_debug = time.time() + 0.5

            now = time.time()
            if now >= next_send:
                pkt = {"s": steer, "t": throttle, "ts": now}
                try:
                    sock.sendto(json.dumps(pkt).encode("utf-8"), (JETSON_HOST, JETSON_PORT))
                    counter += 1
                    # print(f"[NET TX] #{counter} -> {JETSON_HOST}:{JETSON_PORT} s={steer:+.2f} t={throttle:+.2f} ts={now:.2f}")
                except Exception as e:
                    print(f"send error: {e}")
                next_send = now + interval

            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

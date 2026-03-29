#!/usr/bin/env python3
"""
Standalone SLAM smoke test.

Examples:
  python3 client_api/test_slam.py --camera realsense --show
  python3 client_api/test_slam.py --run-dir jetracer/train/runs_rgb_depth/run_20260325_195709 --show
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

try:
    from .hardware import get_camera
    from .slam import VisualSlamSystem
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from hardware import get_camera
    from slam import VisualSlamSystem


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone SLAM smoke test")
    parser.add_argument("--camera", choices=["realsense", "opencv"], default="realsense", help="Camera type for live testing")
    parser.add_argument("--index", type=int, default=0, help="OpenCV camera index when using --camera opencv")
    parser.add_argument("--width", type=int, default=640, help="Live camera width")
    parser.add_argument("--height", type=int, default=480, help="Live camera height")
    parser.add_argument("--fps", type=int, default=15, help="Live camera FPS")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth for live camera testing")
    parser.add_argument("--throttle", type=float, default=0.0, help="Throttle value passed into SLAM for live tests")
    parser.add_argument("--run-dir", type=str, help="Replay a recorded run directory instead of using a live camera")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between replayed frames in seconds")
    parser.add_argument("--print-every", type=int, default=10, help="Print pose every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit, 0 means unlimited")
    parser.add_argument("--show", action="store_true", help="Show the live/replay frame and SLAM map windows")
    return parser


def draw_pose_overlay(frame_bgr, pose, frame_idx):
    overlay = frame_bgr.copy()
    lines = [
        f"Frame: {frame_idx}",
        f"X: {pose['x']:.3f} m",
        f"Y: {pose['y']:.3f} m",
        f"Theta: {pose['theta']:.3f} rad",
    ]

    for idx, text in enumerate(lines):
        y = 24 + idx * 24
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return overlay


def draw_map(pose, size=500, scale=60.0):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)

    cv2.line(canvas, (0, center[1]), (size, center[1]), (40, 40, 40), 1)
    cv2.line(canvas, (center[0], 0), (center[0], size), (40, 40, 40), 1)

    trajectory = pose.get("trajectory") or []
    if len(trajectory) >= 2:
        pts = []
        for x, y in trajectory:
            px = int(center[0] + x * scale)
            py = int(center[1] - y * scale)
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 2)

    x = int(center[0] + pose["x"] * scale)
    y = int(center[1] - pose["y"] * scale)
    cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    arrow_len = 20
    dx = int(np.cos(pose["theta"]) * arrow_len)
    dy = int(np.sin(pose["theta"]) * arrow_len)
    cv2.arrowedLine(canvas, (x, y), (x + dx, y - dy), (0, 255, 0), 2, tipLength=0.3)

    cv2.putText(canvas, "SLAM Map", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resolve_rgb_path(run_dir, row):
    rgb_path = row.get("rgb_path")
    if not rgb_path:
        return None

    if os.path.isabs(rgb_path):
        return rgb_path

    candidate = os.path.join(run_dir, os.path.basename(rgb_path))
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(os.getcwd(), rgb_path)
    if os.path.exists(candidate):
        return candidate

    return None


def replay_run(run_dir):
    csv_path = os.path.join(run_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dataset.csv not found in {run_dir}")

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rgb_path = resolve_rgb_path(run_dir, row)
            if rgb_path is None:
                continue

            frame_bgr = cv2.imread(rgb_path)
            if frame_bgr is None:
                continue

            throttle = 0.0
            try:
                throttle = max(float(row.get("throttle_norm", 0.0) or 0.0), 0.0)
            except ValueError:
                throttle = 0.0

            yield frame_bgr, None, None, throttle


def live_frames(args):
    cam_cfg = {
        "type": args.camera,
        "index": args.index,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    camera = get_camera(cam_cfg, enable_depth=not args.no_depth)

    try:
        while True:
            frame_color, frame_depth, imu_data = camera.read()
            if frame_color is None:
                time.sleep(0.01)
                continue
            yield frame_color, frame_depth, imu_data, args.throttle
    finally:
        camera.release()


def main():
    args = build_parser().parse_args()

    frame_iter = replay_run(args.run_dir) if args.run_dir else live_frames(args)

    slam = None
    frame_idx = 0
    last_pose = None

    try:
        for frame_color, frame_depth, imu_data, throttle in frame_iter:
            if slam is None:
                h, w = frame_color.shape[:2]
                slam = VisualSlamSystem(width=w, height=h)
                print(f"[SLAM] Initialized for {w}x{h}")

            pose = slam.update(frame_color, depth_map=frame_depth, throttle_val=throttle, imu_data=imu_data)
            last_pose = pose

            if frame_idx % max(1, args.print_every) == 0:
                print(
                    f"[{frame_idx:05d}] x={pose['x']:.3f} "
                    f"y={pose['y']:.3f} theta={pose['theta']:.3f}"
                )

            if args.show:
                if args.run_dir:
                    frame_bgr = frame_color
                else:
                    # Camera backends in this project typically return RGB frames.
                    frame_bgr = cv2.cvtColor(frame_color, cv2.COLOR_RGB2BGR)

                cv2.imshow("SLAM View", draw_pose_overlay(frame_bgr, pose, frame_idx))
                cv2.imshow("SLAM Map", draw_map(pose))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            if args.run_dir and args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()

    if last_pose is not None:
        print(
            f"[SLAM] Final pose: x={last_pose['x']:.3f} "
            f"y={last_pose['y']:.3f} theta={last_pose['theta']:.3f}"
        )
    else:
        print("[SLAM] No frames were processed.")


if __name__ == "__main__":
    main()

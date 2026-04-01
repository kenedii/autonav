import math
import os
import threading
import time
import logging
import sys

import cv2
import numpy as np

try:
    from .hardware import PCA9685, build_sensor_rig, empty_sensor_snapshot, get_system_specs
    from .models import AutonomousDriver, ObjectDetector
    from .slam import VisualSlamSystem
    from .mission import MissionManager
    from .tag_detector import AprilTagDetector
except ImportError:
    from hardware import PCA9685, build_sensor_rig, empty_sensor_snapshot, get_system_specs
    from models import AutonomousDriver, ObjectDetector
    from slam import VisualSlamSystem
    from mission import MissionManager
    from tag_detector import AprilTagDetector

try:
    from preprocess_utils import infer_preprocess_profile
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from preprocess_utils import infer_preprocess_profile


logger = logging.getLogger("CarClient")


class CarClient(object):
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
            "specs": {},
            "mission": {},
            "sensors": empty_sensor_snapshot(),
        }

        try:
            self.state["specs"] = get_system_specs()
        except Exception as exc:
            logger.warning("Failed to get system specs: %s", exc)

        self.sensor_rig = None
        self.pca = None
        self.control_model = None
        self.detection_model = None
        self.slam = None
        self.action_loop = ["control", "api"]

        self.target_dest = None
        self.nav_kp = 2.0

        self.thread = None
        self.lock = threading.Lock()
        self.latest_preview_jpeg = None
        self.latest_preview_lock = threading.Lock()
        self.mission_manager = MissionManager({"enabled": False})
        self.tag_detector = None
        self.front_obstacle_distance_m = None
        self.last_tag_detections = []
        self.last_depth_frame_ts = None
        self._last_mission_log_key = None
        self._last_camera_warning_ts = 0.0

        self.STEERING_CHANNEL = 0
        self.THROTTLE_CHANNEL = 1
        self.STEERING_CENTER = 1500
        self.THROTTLE_CENTER = 1500
        self.THROTTLE_MAX = 1900
        self.THROTTLE_MIN = 1200

        self.fixed_throttle = 0.22

        self.state["mission"] = self.mission_manager.snapshot()

        if os.path.exists("best_model.pth"):
            logger.info("Auto-loading local 'best_model.pth'...")
            default_config = {
                "device": "cuda",
                "architecture": "resnet101",
                "cameras": [{"type": "realsense", "width": 640, "height": 480, "fps": 15}],
                "control_model_type": "pytorch",
                "control_model": "best_model.pth",
                "detection_model": "yolov8n.pt",
                "action_loop": ["control", "detection"],
            }
            trt_path = "/home/jetson/jetracer_run/checkpoints/checkpoints/model_7_resnet101/best_model_trt.pth"
            if os.path.exists(trt_path):
                logger.info("Found optimized TensorRT model at %s", trt_path)
                default_config["control_model_type"] = "tensorrt"
                default_config["control_model"] = trt_path
            try:
                self.configure(default_config)
            except Exception as exc:
                logger.error("Auto-config failed: %s", exc)

    def set_throttle_mode(self, mode, value=None):
        with self.lock:
            self.config["throttle_mode"] = mode
            if value is not None:
                self.fixed_throttle = float(value)
            logger.info("Throttle mode set to %s (val=%s)", mode, value)

    def configure(self, config):
        config = dict(config or {})
        previous_thread = self.thread
        if self.running:
            self.running = False
        if previous_thread and previous_thread.is_alive() and previous_thread is not threading.current_thread():
            previous_thread.join(timeout=2.0)

        with self.lock:
            if config.get("control_model"):
                config["control_model"] = os.path.expanduser(config["control_model"])
                if config["control_model"].startswith("jetson:"):
                    config["control_model"] = config["control_model"].replace("jetson:", "", 1)
            if config.get("detection_model"):
                config["detection_model"] = os.path.expanduser(config["detection_model"])
            if config.get("fixed_throttle_value") is not None:
                self.fixed_throttle = float(config["fixed_throttle_value"])
            config["preprocess_profile"] = infer_preprocess_profile(
                config.get("cameras"),
                config.get("preprocess_profile"),
            )

            self.action_loop = config.get("action_loop") or ["control", "api"]

            try:
                specs = get_system_specs(config.get("cameras", []))
                if config.get("architecture"):
                    specs["resnet_version"] = config["architecture"]
                if config.get("detection_model"):
                    specs["yolo_version"] = os.path.basename(config["detection_model"])
                self.state["specs"] = specs
            except Exception as exc:
                logger.warning("Error updating specs: %s", exc)

            if self.sensor_rig:
                try:
                    self.sensor_rig.release()
                except Exception:
                    pass
                self.sensor_rig = None

            self.config = config
            self.paused = False
            self.pause_until = 0
            self.state["detections"] = []
            self.state["last_action"] = None
            self.state["location"] = None
            self.front_obstacle_distance_m = None
            self.last_tag_detections = []
            self.last_depth_frame_ts = None
            self._last_mission_log_key = None
            self.state["sensors"] = empty_sensor_snapshot()
            with self.latest_preview_lock:
                self.latest_preview_jpeg = None

            logger.info("Configuring CarClient with model: %s", config.get("control_model"))

            self.mission_manager = MissionManager(config.get("mission") or {})
            self.state["mission"] = self.mission_manager.snapshot()

            try:
                if self.pca is None:
                    self.pca = PCA9685()
                self.pca.set_us(self.STEERING_CHANNEL, self.STEERING_CENTER)
                self.pca.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
            except Exception as exc:
                logger.error("Failed to init PCA9685 (Mocking): %s", exc)
                self.pca = None

            need_depth = (
                ("detection" in self.action_loop) or
                ("slam" in self.action_loop) or
                bool(self.mission_manager.depth_stop.get("enabled"))
            )
            try:
                self.sensor_rig = build_sensor_rig(config.get("cameras") or [], enable_depth=need_depth)
                self.state["sensors"] = self.sensor_rig.snapshot()
                if not self.sensor_rig.primary_rgb_camera:
                    logger.warning("Primary RGB camera initialization returned None")
            except Exception as exc:
                logger.error("Failed to init sensor rig: %s", exc)
                self.sensor_rig = None
                self.state["sensors"] = empty_sensor_snapshot()

            try:
                self.control_model = None
                if config.get("control_model"):
                    self.control_model = AutonomousDriver(config)
                self.detection_model = None
                if "detection" in self.action_loop and config.get("detection_model"):
                    self.detection_model = ObjectDetector(config)
                if "slam" in self.action_loop:
                    cam_cfg = (self.state["sensors"].get("primary_rgb") or {}) if isinstance(self.state.get("sensors"), dict) else {}
                    self.slam = VisualSlamSystem(
                        width=cam_cfg.get("width", 640),
                        height=cam_cfg.get("height", 480),
                    )
                else:
                    self.slam = None
            except Exception as exc:
                logger.error("Failed to init Models: %s", exc)
                self.control_model = None
                self.detection_model = None
                self.slam = None

            self.tag_detector = AprilTagDetector()
            if self.mission_manager.enabled:
                self.mission_manager.set_tag_detector_status(self.tag_detector.status)
            else:
                self.mission_manager.set_tag_detector_status("disabled")
            self.mission_manager.set_control_model_status(
                "available" if self.control_model is not None else "unavailable"
            )
            self.mission_manager.set_depth_status(
                "unknown" if self.mission_manager.depth_stop.get("enabled") else "disabled"
            )
            self._publish_mission_state()

    def start_logic(self):
        self.mission_manager.start()
        self._publish_mission_state()
        if self.running:
            logger.info("Car logic already running; mission state refreshed.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Car logic started.")

    def stop_logic(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        self._neutralize_outputs()

        if self.sensor_rig:
            self.sensor_rig.release()
            self.sensor_rig = None
            self.state["sensors"] = empty_sensor_snapshot()

        self.mission_manager.stop("operator_stop")
        self.state["last_action"] = {"steer": 0.0, "throttle": 0.0}
        self._publish_mission_state()

    def pause(self, duration=None):
        self.paused = True
        if self.pca:
            self.pca.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
        if duration:
            self.pause_until = time.time() + duration
            logger.info("Paused for %s seconds", duration)
        else:
            self.pause_until = 0
            logger.info("Paused indefinitely")

    def resume(self):
        self.paused = False
        self.pause_until = 0
        logger.info("Resumed")

    def _neutralize_outputs(self):
        if not self.pca:
            return
        self.pca.set_us(self.THROTTLE_CHANNEL, self.THROTTLE_CENTER)
        self.pca.set_us(self.STEERING_CHANNEL, self.STEERING_CENTER)

    def _apply_runtime_stop(self, reason, now_ts=None):
        if now_ts is None:
            now_ts = time.time()
        self.mission_manager.stop(reason, now_ts)
        self.state["last_action"] = {"steer": 0.0, "throttle": 0.0}
        self._neutralize_outputs()
        self._publish_mission_state()

    def get_latest_preview_jpeg(self):
        with self.latest_preview_lock:
            return self.latest_preview_jpeg

    def _estimate_front_obstacle_distance_m(self, depth_frame):
        if depth_frame is None:
            return None

        roi_cfg = self.mission_manager.depth_stop.get("roi") or {}
        height, width = depth_frame.shape[:2]
        x1 = max(0, min(width - 1, int(width * roi_cfg.get("x", 0.35))))
        y1 = max(0, min(height - 1, int(height * roi_cfg.get("y", 0.35))))
        x2 = max(x1 + 1, min(width, int(width * (roi_cfg.get("x", 0.35) + roi_cfg.get("w", 0.30)))))
        y2 = max(y1 + 1, min(height, int(height * (roi_cfg.get("y", 0.35) + roi_cfg.get("h", 0.30)))))

        crop = depth_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        valid_depths = crop[crop > 0]
        if valid_depths.size == 0:
            return None

        median_mm = float(np.median(valid_depths))
        return median_mm / 1000.0

    def _update_preview(self, frame_bgr, mission_state, tag_detections, obstacle_distance_m, last_action):
        if frame_bgr is None:
            return

        preview = frame_bgr.copy()
        overlay_lines = [
            "Mission: %s" % mission_state.get("state", "IDLE"),
            "Route: %s" % mission_state.get("route_name", "expo_route"),
            "Tag detector: %s" % mission_state.get("tag_detector_status", "unknown"),
        ]
        if mission_state.get("stop_reason"):
            overlay_lines.append("Stop: %s" % mission_state["stop_reason"])
        if obstacle_distance_m is not None:
            overlay_lines.append("Obstacle: %.2fm" % obstacle_distance_m)
        last_tag_id = mission_state.get("last_tag_id")
        if last_tag_id is not None:
            overlay_lines.append("Last tag: %s" % last_tag_id)
        if last_action:
            overlay_lines.append(
                "Steer %.2f  Throttle %.2f" % (
                    float(last_action.get("steer", 0.0)),
                    float(last_action.get("throttle", 0.0)),
                )
            )

        y = 24
        for line in overlay_lines:
            cv2.putText(
                preview,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (32, 255, 32),
                2,
                cv2.LINE_AA,
            )
            y += 24

        for detection in tag_detections or []:
            corners = detection.get("corners") or []
            if len(corners) >= 4:
                points = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [points], True, (0, 200, 255), 2)
            center = detection.get("center") or [0, 0]
            cv2.putText(
                preview,
                "Tag %s" % detection.get("id"),
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        with self.latest_preview_lock:
            self.latest_preview_jpeg = encoded.tobytes()

    def _publish_mission_state(self):
        snapshot = self.mission_manager.snapshot()
        self.state["mission"] = snapshot
        key = (
            snapshot.get("enabled"),
            snapshot.get("state"),
            snapshot.get("stop_reason"),
            snapshot.get("start_tag_seen"),
            snapshot.get("checkpoint_seen"),
            snapshot.get("goal_seen"),
            snapshot.get("expected_next_tag"),
            snapshot.get("tag_detector_status"),
            snapshot.get("control_model_status"),
            snapshot.get("depth_status"),
        )
        if key != self._last_mission_log_key:
            logger.info(
                "Mission state=%s stop=%s next=%s detector=%s depth=%s",
                snapshot.get("state"),
                snapshot.get("stop_reason"),
                snapshot.get("expected_next_tag"),
                snapshot.get("tag_detector_status"),
                snapshot.get("depth_status"),
            )
            self._last_mission_log_key = key
        return snapshot

    def _step_once(self, frame_count, now_ts=None):
        if now_ts is None:
            now_ts = time.time()

        sensor_packet = {}
        if self.sensor_rig is not None:
            sensor_packet = self.sensor_rig.read()
            self.state["sensors"] = sensor_packet.get("sensor_snapshot") or self.sensor_rig.snapshot()
        else:
            self.state["sensors"] = empty_sensor_snapshot()

        frame_color = sensor_packet.get("primary_rgb")
        frame_depth = sensor_packet.get("depth")
        imu_data = sensor_packet.get("imu") or {}
        depth_aligned_to_primary = bool(sensor_packet.get("depth_aligned_to_primary"))
        primary_sensor_state = (self.state.get("sensors") or {}).get("primary_rgb") or {}
        primary_configured = bool(primary_sensor_state.get("configured"))

        if frame_color is None:
            if now_ts - self._last_camera_warning_ts > 2.0:
                if self.sensor_rig is None or not primary_configured:
                    logger.warning("No primary RGB camera configured! Waiting for configuration...")
                else:
                    logger.warning("Primary RGB camera unavailable; neutralizing outputs.")
                self._last_camera_warning_ts = now_ts
            self.state["detections"] = []
            self.front_obstacle_distance_m = None
            self._apply_runtime_stop("primary_rgb_unavailable", now_ts)
            time.sleep(0.05 if primary_configured else 2.0)
            return False

        if frame_depth is not None:
            self.last_depth_frame_ts = now_ts

        frame_bgr = cv2.cvtColor(frame_color, cv2.COLOR_RGB2BGR)
        detections = []
        tag_detections = []
        override_steer = None
        force_zero_throttle = False

        slam_depth = frame_depth if depth_aligned_to_primary else None
        if self.slam is not None:
            pose = self.slam.update(
                frame_color,
                slam_depth,
                throttle_val=self.fixed_throttle,
                imu_data=imu_data,
            )
            if isinstance(pose, dict):
                pose = dict(pose)
                if imu_data:
                    pose["imu"] = imu_data
                self.state["location"] = pose
                if self.target_dest:
                    dx = self.target_dest[0] - pose["x"]
                    dy = self.target_dest[1] - pose["y"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 0.2:
                        logger.info("Nav: Reached destination %s", self.target_dest)
                        self.target_dest = None
                        override_steer = 0.0
                    else:
                        target_theta = math.atan2(dy, dx)
                        error = target_theta - pose["theta"]
                        error = (error + math.pi) % (2 * math.pi) - math.pi
                        override_steer = float(np.clip(error * self.nav_kp, -1.0, 1.0))
        elif imu_data:
            loc = self.state.get("location") or {}
            if isinstance(loc, dict):
                loc = dict(loc)
            else:
                loc = {}
            loc["imu"] = imu_data
            self.state["location"] = loc

        obstacle_distance = None
        if self.mission_manager.depth_stop.get("enabled"):
            obstacle_distance = self._estimate_front_obstacle_distance_m(frame_depth)
            self.front_obstacle_distance_m = obstacle_distance
            if obstacle_distance is not None:
                self.mission_manager.update_obstacle(obstacle_distance, now_ts)
            else:
                self.mission_manager.update_obstacle(None, now_ts)
                if self.last_depth_frame_ts is None or (now_ts - self.last_depth_frame_ts) > 0.5:
                    self.mission_manager.stop("depth_unavailable", now_ts)
                    force_zero_throttle = True
        else:
            self.front_obstacle_distance_m = None
            self.mission_manager.set_depth_status("disabled")

        if self.tag_detector is not None:
            self.mission_manager.set_tag_detector_status(
                self.tag_detector.status if self.mission_manager.enabled else "disabled"
            )
            tag_every = self.mission_manager.tag_detect_every_n_frames
            if self.mission_manager.enabled and self.tag_detector.available and frame_count % tag_every == 0:
                tag_detections = self.tag_detector.detect(frame_bgr)
                self.last_tag_detections = tag_detections
            elif frame_count % max(1, self.mission_manager.tag_detect_every_n_frames) == 0:
                self.last_tag_detections = []
            tag_detections = list(self.last_tag_detections)
            detected_tag_ids = [tag["id"] for tag in tag_detections]
            for tag_id in self.mission_manager.consume_tags(detected_tag_ids, now_ts):
                logger.info("Mission tag seen: %s", tag_id)

        if "detection" in self.action_loop and self.detection_model is not None:
            detections = self.detection_model.detect(frame_color) or []
            if depth_aligned_to_primary and frame_depth is not None:
                for detection in detections:
                    bbox = detection.get("bbox") or []
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_depth.shape[1], x2)
                    y2 = min(frame_depth.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame_depth[y1:y2, x1:x2]
                    valid_depths = crop[crop > 0]
                    if valid_depths.size > 0:
                        detection["distance"] = float(np.mean(valid_depths)) / 1000.0
        self.state["detections"] = detections

        if self.control_model is None:
            self.mission_manager.set_control_model_status("unavailable")
            self.mission_manager.stop("model_unavailable", now_ts)
            steer_norm = 0.0
            force_zero_throttle = True
        else:
            self.mission_manager.set_control_model_status("available")
            if override_steer is not None:
                steer_norm = float(override_steer)
            else:
                steer_norm = float(self.control_model.predict(frame_color))

        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
        current_throttle = self.mission_manager.compute_throttle(self.fixed_throttle, abs(steer_norm))
        if force_zero_throttle:
            current_throttle = 0.0

        pulse = int(self.STEERING_CENTER + (steer_norm * 500))
        pulse = int(np.clip(pulse, 1000, 2000))
        throttle_us = self.THROTTLE_CENTER + int(
            current_throttle * (self.THROTTLE_MAX - self.THROTTLE_CENTER)
        )

        if self.pca:
            self.pca.set_us(self.STEERING_CHANNEL, pulse)
            self.pca.set_us(self.THROTTLE_CHANNEL, throttle_us)

        self.state["last_action"] = {
            "steer": float(steer_norm),
            "throttle": float(current_throttle),
        }

        mission_snapshot = self._publish_mission_state()
        self._update_preview(
            frame_bgr,
            mission_snapshot,
            tag_detections,
            obstacle_distance,
            self.state["last_action"],
        )
        return True

    def _loop(self):
        frame_count = 0
        fps_counter = 0
        last_fps_time = time.time()

        if self.pca:
            time.sleep(1.0)

        while self.running:
            now_ts = time.time()

            if self.paused:
                if self.pause_until > 0 and now_ts > self.pause_until:
                    self.resume()
                else:
                    time.sleep(0.1)
                    continue

            if self._step_once(frame_count, now_ts=now_ts):
                fps_counter += 1
                frame_count += 1
            if now_ts - last_fps_time >= 1.0:
                self.state["fps"] = fps_counter
                fps_counter = 0
                last_fps_time = now_ts

            time.sleep(0.001)


car = CarClient()

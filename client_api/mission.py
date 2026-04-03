import threading
import time


DEFAULT_ROUTE_NAME = "expo_route"
DEFAULT_TAG_IDS = {
    "start_home": 10,
    "checkpoint": 20,
    "goal": 30,
}
DEFAULT_TAG_COOLDOWN_S = 1.25
DEFAULT_TAG_DETECT_EVERY_N_FRAMES = 3
DEFAULT_DEPTH_ROI = {
    "x": 0.35,
    "y": 0.35,
    "w": 0.30,
    "h": 0.30,
}

MISSION_IDLE = "IDLE"
MISSION_RUNNING = "RUNNING"
MISSION_CHECKPOINT_SEEN = "CHECKPOINT_SEEN"
MISSION_APPROACH_GOAL = "APPROACH_GOAL"
MISSION_COMPLETE = "COMPLETE"
MISSION_FAULT_STOP = "FAULT_STOP"


def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


class MissionManager(object):
    def __init__(self, mission_config=None):
        self._lock = threading.RLock()
        self._recent_tag_times = {}

        self.enabled = False
        self.route_name = DEFAULT_ROUTE_NAME
        self.tag_ids = dict(DEFAULT_TAG_IDS)
        self.tag_cooldown_s = DEFAULT_TAG_COOLDOWN_S
        self.tag_detect_every_n_frames = DEFAULT_TAG_DETECT_EVERY_N_FRAMES
        self.depth_stop = {
            "enabled": False,
            "threshold_m": 0.60,
            "roi": dict(DEFAULT_DEPTH_ROI),
        }

        self.state = MISSION_IDLE
        self.stop_reason = None
        self.start_tag_seen = False
        self.checkpoint_seen = False
        self.goal_seen = False
        self.expected_next_tag = None
        self.last_tag_id = None
        self.last_tag_seen_ts = None
        self.last_checkpoint_seen_ts = None
        self.last_goal_seen_ts = None
        self.goal_approach_since = None
        self.obstacle_distance_m = None
        self.state_changed_at = time.time()
        self.tag_detector_status = "disabled"
        self.control_model_status = "unknown"
        self.depth_status = "disabled"

        self.configure(mission_config or {})

    def configure(self, mission_config):
        if not isinstance(mission_config, dict):
            mission_config = {}

        tag_ids = mission_config.get("tag_ids") or {}
        depth_stop = mission_config.get("depth_stop") or {}
        roi = depth_stop.get("roi") or {}

        with self._lock:
            self.enabled = bool(mission_config.get("enabled", False))
            self.route_name = mission_config.get("route_name") or DEFAULT_ROUTE_NAME
            self.tag_ids = {
                "start_home": _coerce_int(tag_ids.get("start_home"), DEFAULT_TAG_IDS["start_home"]),
                "checkpoint": _coerce_int(tag_ids.get("checkpoint"), DEFAULT_TAG_IDS["checkpoint"]),
                "goal": _coerce_int(tag_ids.get("goal"), DEFAULT_TAG_IDS["goal"]),
            }
            self.tag_cooldown_s = _coerce_float(
                mission_config.get("tag_cooldown_s"),
                DEFAULT_TAG_COOLDOWN_S,
            )
            self.tag_detect_every_n_frames = max(
                1,
                _coerce_int(
                    mission_config.get("tag_detect_every_n_frames"),
                    DEFAULT_TAG_DETECT_EVERY_N_FRAMES,
                ),
            )
            self.depth_stop = {
                "enabled": bool(depth_stop.get("enabled", self.enabled)),
                "threshold_m": _coerce_float(depth_stop.get("threshold_m"), 0.60),
                "roi": {
                    "x": _coerce_float(roi.get("x"), DEFAULT_DEPTH_ROI["x"]),
                    "y": _coerce_float(roi.get("y"), DEFAULT_DEPTH_ROI["y"]),
                    "w": _coerce_float(roi.get("w"), DEFAULT_DEPTH_ROI["w"]),
                    "h": _coerce_float(roi.get("h"), DEFAULT_DEPTH_ROI["h"]),
                },
            }
            if not self.enabled:
                self.depth_stop["enabled"] = False

            self._reset_unlocked()
            self.tag_detector_status = "disabled" if not self.enabled else "unavailable"
            self.control_model_status = "unknown"
            self.depth_status = "disabled" if not self.depth_stop["enabled"] else "unknown"

    def reset(self):
        with self._lock:
            self._reset_unlocked()

    def start(self, now_ts=None):
        with self._lock:
            if not self.enabled:
                return
            if now_ts is None:
                now_ts = time.time()
            if self.state == MISSION_COMPLETE or self.stop_reason == "operator_stop":
                self._reset_unlocked(now_ts)

            self.stop_reason = None
            if self.goal_seen and self.state != MISSION_COMPLETE:
                self.state = MISSION_APPROACH_GOAL
                self.expected_next_tag = None
            elif self.checkpoint_seen:
                self.state = MISSION_CHECKPOINT_SEEN
                self.expected_next_tag = self.tag_ids["goal"]
            else:
                self.state = MISSION_RUNNING
                self.expected_next_tag = self.tag_ids["checkpoint"]
            self.state_changed_at = now_ts

    def stop(self, reason=None, now_ts=None):
        with self._lock:
            if now_ts is None:
                now_ts = time.time()

            if reason in (None, "operator_stop"):
                self._reset_unlocked(now_ts)
                self.stop_reason = "operator_stop"
                return

            if reason == "goal_reached":
                self.state = MISSION_COMPLETE
                self.stop_reason = "goal_reached"
                self.goal_seen = True
                self.expected_next_tag = None
                self.goal_approach_since = None
                self.state_changed_at = now_ts
                return

            self.state = MISSION_FAULT_STOP
            self.stop_reason = reason
            self.state_changed_at = now_ts

    def consume_tags(self, tag_ids, now_ts=None):
        if now_ts is None:
            now_ts = time.time()

        consumed = []
        with self._lock:
            if not self.enabled or self.state in (MISSION_COMPLETE, MISSION_FAULT_STOP):
                return consumed

            for raw_tag in tag_ids or []:
                tag_id = _coerce_int(raw_tag, None)
                if tag_id is None:
                    continue

                last_seen = self._recent_tag_times.get(tag_id)
                if last_seen is not None and (now_ts - last_seen) < self.tag_cooldown_s:
                    continue

                self._recent_tag_times[tag_id] = now_ts
                self.last_tag_id = tag_id
                self.last_tag_seen_ts = now_ts

                if tag_id == self.tag_ids["start_home"]:
                    self.start_tag_seen = True
                    if not self.checkpoint_seen:
                        self.expected_next_tag = self.tag_ids["checkpoint"]
                    consumed.append(tag_id)
                    continue

                if tag_id == self.tag_ids["checkpoint"]:
                    if self.checkpoint_seen:
                        consumed.append(tag_id)
                        continue
                    self.checkpoint_seen = True
                    self.last_checkpoint_seen_ts = now_ts
                    self.state = MISSION_CHECKPOINT_SEEN
                    self.expected_next_tag = self.tag_ids["goal"]
                    self.state_changed_at = now_ts
                    consumed.append(tag_id)
                    continue

                if tag_id == self.tag_ids["goal"]:
                    if not self.checkpoint_seen:
                        continue
                    if self.goal_seen:
                        consumed.append(tag_id)
                        continue
                    self.goal_seen = True
                    self.last_goal_seen_ts = now_ts
                    self.goal_approach_since = now_ts
                    self.state = MISSION_APPROACH_GOAL
                    self.expected_next_tag = None
                    self.state_changed_at = now_ts
                    consumed.append(tag_id)

            return consumed

    def update_obstacle(self, distance_m, now_ts=None):
        with self._lock:
            if now_ts is None:
                now_ts = time.time()

            if distance_m is None:
                self.obstacle_distance_m = None
                if self.depth_stop["enabled"]:
                    self.depth_status = "unavailable"
                return False

            distance = _coerce_float(distance_m, None)
            if distance is None or distance <= 0:
                self.obstacle_distance_m = None
                if self.depth_stop["enabled"]:
                    self.depth_status = "unavailable"
                return False

            self.obstacle_distance_m = distance
            if not self.depth_stop["enabled"]:
                self.depth_status = "disabled"
                return False

            self.depth_status = "available"
            if self.enabled and distance < self.depth_stop["threshold_m"] and self.state != MISSION_COMPLETE:
                self.state = MISSION_FAULT_STOP
                self.stop_reason = "obstacle"
                self.state_changed_at = now_ts
                return True
            return False

    def compute_throttle(self, base_throttle, steer_abs):
        with self._lock:
            throttle = _coerce_float(base_throttle, 0.0)
            steer_abs = abs(_coerce_float(steer_abs, 0.0))

            if self.state in (MISSION_FAULT_STOP, MISSION_COMPLETE):
                return 0.0

            if not self.enabled:
                return max(0.0, min(throttle, 1.0))

            if self.state == MISSION_IDLE:
                return 0.0

            if steer_abs > 0.65:
                throttle *= 0.70
            elif steer_abs > 0.35:
                throttle *= 0.85

            if self.state == MISSION_APPROACH_GOAL:
                goal_cap = min(_coerce_float(base_throttle, 0.0) * 0.5, 0.12)
                throttle = min(throttle, goal_cap)
                if self.goal_approach_since is not None and (time.time() - self.goal_approach_since) >= 0.75:
                    self.state = MISSION_COMPLETE
                    self.stop_reason = "goal_reached"
                    self.goal_approach_since = None
                    self.expected_next_tag = None
                    self.state_changed_at = time.time()
                    return 0.0

            return max(0.0, min(throttle, 1.0))

    def set_tag_detector_status(self, status):
        with self._lock:
            if not self.enabled:
                self.tag_detector_status = "disabled"
            else:
                self.tag_detector_status = status or "unavailable"

    def set_control_model_status(self, status):
        with self._lock:
            self.control_model_status = status or "unknown"

    def set_depth_status(self, status):
        with self._lock:
            if not self.depth_stop["enabled"]:
                self.depth_status = "disabled"
            else:
                self.depth_status = status or "unknown"

    def snapshot(self):
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "route_name": self.route_name,
                "state": self.state,
                "stop_reason": self.stop_reason,
                "tag_detector_status": self.tag_detector_status,
                "control_model_status": self.control_model_status,
                "depth_status": self.depth_status,
                "obstacle_distance_m": self.obstacle_distance_m,
                "start_tag_seen": bool(self.start_tag_seen),
                "checkpoint_seen": bool(self.checkpoint_seen),
                "goal_seen": bool(self.goal_seen),
                "expected_next_tag": self.expected_next_tag,
                "last_tag_id": self.last_tag_id,
                "last_tag_seen_ts": self.last_tag_seen_ts,
                "last_checkpoint_seen_ts": self.last_checkpoint_seen_ts,
                "last_goal_seen_ts": self.last_goal_seen_ts,
                "goal_approach_since": self.goal_approach_since,
                "state_changed_at": self.state_changed_at,
                "tag_cooldown_s": self.tag_cooldown_s,
                "tag_detect_every_n_frames": self.tag_detect_every_n_frames,
                "depth_stop": {
                    "enabled": bool(self.depth_stop["enabled"]),
                    "threshold_m": self.depth_stop["threshold_m"],
                    "roi": {
                        "x": self.depth_stop["roi"]["x"],
                        "y": self.depth_stop["roi"]["y"],
                        "w": self.depth_stop["roi"]["w"],
                        "h": self.depth_stop["roi"]["h"],
                    },
                },
                "tag_ids": dict(self.tag_ids),
            }

    def _reset_unlocked(self, now_ts=None):
        if now_ts is None:
            now_ts = time.time()
        self.state = MISSION_IDLE
        self.stop_reason = None
        self.start_tag_seen = False
        self.checkpoint_seen = False
        self.goal_seen = False
        self.expected_next_tag = None if not self.enabled else self.tag_ids["start_home"]
        self.last_tag_id = None
        self.last_tag_seen_ts = None
        self.last_checkpoint_seen_ts = None
        self.last_goal_seen_ts = None
        self.goal_approach_since = None
        self.obstacle_distance_m = None
        self.state_changed_at = now_ts
        self._recent_tag_times = {}

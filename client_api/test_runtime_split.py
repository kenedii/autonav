import sys
import json
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules["smbus2"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torchvision"] = MagicMock()
sys.modules["torchvision.models"] = MagicMock()

import hardware
from car import CarClient
from mission import MissionManager
from preprocess_utils import (
    CAM0_FISHEYE_PREPROCESS_PROFILE,
    LEGACY_PREPROCESS_PROFILE,
    apply_preprocess_profile,
    infer_preprocess_profile,
)


class FakeRig:
    def __init__(self, packet, snapshot):
        self.packet = dict(packet)
        self._snapshot = snapshot
        self.read_calls = []

    def read(self, include_rear=False):
        self.read_calls.append(include_rear)
        packet = dict(self.packet)
        packet["sensor_snapshot"] = self._snapshot
        return packet

    def snapshot(self):
        return self._snapshot

    def release(self):
        return None


class StubCamera:
    def __init__(self, frame=None, depth=None, imu=None):
        self.frame = frame
        self.depth = depth
        self.imu = imu

    def read(self):
        return self.frame, self.depth, self.imu

    def release(self):
        return None


def _sensor_snapshot(primary_configured=True, sidecar_configured=False):
    snapshot = hardware.empty_sensor_snapshot()
    snapshot["primary_rgb"]["configured"] = primary_configured
    snapshot["primary_rgb"]["enabled"] = primary_configured
    snapshot["primary_rgb"]["status"] = "configured" if primary_configured else "disabled"
    snapshot["sidecar_depth_imu"]["configured"] = sidecar_configured
    snapshot["sidecar_depth_imu"]["enabled"] = sidecar_configured
    snapshot["sidecar_depth_imu"]["status"] = "configured" if sidecar_configured else "disabled"
    return snapshot


def _build_client(packet, mission_config=None):
    client = CarClient()
    client.pca = MagicMock()
    client.sensor_rig = FakeRig(packet, _sensor_snapshot(primary_configured=True, sidecar_configured=True))
    client.mission_manager = MissionManager(mission_config or {"enabled": True})
    client.tag_detector = MagicMock()
    client.tag_detector.status = "unavailable"
    client.tag_detector.available = False
    client.tag_detector.detect.return_value = []
    client.control_model = MagicMock()
    client.control_model.predict.return_value = 0.15
    client.detection_model = None
    client.slam = None
    client.action_loop = ["control", "api"]
    return client


def test_infer_preprocess_profile_defaults_to_cam0_for_role_based_csi():
    profile = infer_preprocess_profile(
        [{"role": "primary_rgb", "type": "csi", "sensor_id": 0}],
        None,
    )
    assert profile == CAM0_FISHEYE_PREPROCESS_PROFILE


def test_apply_preprocess_profile_cam0_is_bottom_weighted():
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    frame[:30, :, :] = 10
    frame[30:, :, :] = 220

    legacy = apply_preprocess_profile(frame, LEGACY_PREPROCESS_PROFILE)
    cam0 = apply_preprocess_profile(frame, CAM0_FISHEYE_PREPROCESS_PROFILE)

    assert legacy.shape == (120, 160, 3)
    assert cam0.shape == (120, 160, 3)
    assert cam0.mean() > legacy.mean()


def test_resolve_sensor_role_configs_legacy_realsense_aliases_sidecar():
    resolved = hardware.resolve_sensor_role_configs(
        [{"type": "realsense", "width": 640, "height": 480, "fps": 15}]
    )

    assert resolved["role_based"] is False
    assert resolved["primary_rgb"] is resolved["sidecar_depth_imu"]


def test_step_once_stops_safely_when_primary_rgb_disappears():
    client = _build_client({"primary_rgb": None, "depth": None, "imu": {}})

    processed = client._step_once(frame_count=0, now_ts=100.0)

    assert processed is False
    assert client.state["mission"]["stop_reason"] == "primary_rgb_unavailable"
    assert client.state["last_action"] == {"steer": 0.0, "throttle": 0.0}
    client.pca.set_us.assert_any_call(client.THROTTLE_CHANNEL, client.THROTTLE_CENTER)
    client.pca.set_us.assert_any_call(client.STEERING_CHANNEL, client.STEERING_CENTER)


def test_step_once_stops_safely_when_depth_is_missing():
    mission_config = {
        "enabled": True,
        "depth_stop": {"enabled": True, "threshold_m": 0.6},
    }
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    client = _build_client(
        {"primary_rgb": frame, "depth": None, "imu": {}, "depth_aligned_to_primary": False},
        mission_config=mission_config,
    )

    processed = client._step_once(frame_count=0, now_ts=100.0)

    assert processed is True
    assert client.state["mission"]["stop_reason"] == "depth_unavailable"
    assert client.state["last_action"]["throttle"] == 0.0
    assert client.sensor_rig.read_calls == [False]
    assert client.get_latest_preview_jpeg() is not None


def test_missing_imu_is_visible_without_stopping_forward_control():
    mission_config = {
        "enabled": True,
        "depth_stop": {"enabled": True, "threshold_m": 0.6},
    }
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 1000, dtype=np.uint16)
    snapshot = _sensor_snapshot(primary_configured=True, sidecar_configured=True)
    snapshot["sidecar_depth_imu"]["imu"] = "unavailable"
    snapshot["sidecar_depth_imu"]["imu_status"] = "unavailable"
    snapshot["sidecar_depth_imu"]["imu_frame_age_ms"] = None
    snapshot["imu"] = {
        "role": "imu",
        "source": "realsense",
        "status": "unavailable",
        "frame_age_ms": None,
    }

    client = CarClient()
    client.pca = MagicMock()
    client.sensor_rig = FakeRig(
        {"primary_rgb": frame, "depth": depth, "imu": None},
        snapshot,
    )
    client.mission_manager = MissionManager(mission_config)
    client.tag_detector = MagicMock()
    client.tag_detector.status = "unavailable"
    client.tag_detector.available = False
    client.tag_detector.detect.return_value = []
    client.control_model = MagicMock()
    client.control_model.predict.return_value = 0.2
    client.detection_model = None
    client.slam = None
    client.action_loop = ["control", "api"]
    client.mission_manager.start()

    processed = client._step_once(frame_count=0, now_ts=100.0)

    assert processed is True
    assert client.state["mission"]["stop_reason"] is None
    assert client.state["last_action"]["throttle"] > 0.0
    assert client.state["sensors"]["sidecar_depth_imu"]["imu"] == "unavailable"
    assert client.state["sensors"]["sidecar_depth_imu"]["imu_status"] == "unavailable"
    assert client.state["sensors"]["imu"]["status"] == "unavailable"
    assert client.sensor_rig.read_calls == [False]


def test_missing_cam1_does_not_affect_forward_control():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = np.full((120, 160), 1000, dtype=np.uint16)
    snapshot = _sensor_snapshot(primary_configured=True, sidecar_configured=True)
    snapshot["rear_preview"]["enabled"] = False
    snapshot["rear_preview"]["healthy"] = False
    snapshot["rear_preview"]["status"] = "disabled"
    snapshot["rear_preview"]["frame_age_ms"] = None
    snapshot["rear"] = {
        "role": "rear",
        "source": "cam1",
        "status": "disabled",
        "frame_age_ms": None,
    }

    client = CarClient()
    client.pca = MagicMock()
    client.sensor_rig = FakeRig(
        {"primary_rgb": frame, "depth": depth, "imu": {}},
        snapshot,
    )
    client.mission_manager = MissionManager({"enabled": True})
    client.tag_detector = MagicMock()
    client.tag_detector.status = "unavailable"
    client.tag_detector.available = False
    client.tag_detector.detect.return_value = []
    client.control_model = MagicMock()
    client.control_model.predict.return_value = 0.25
    client.detection_model = None
    client.slam = None
    client.action_loop = ["control", "api"]
    client.mission_manager.start()

    processed = client._step_once(frame_count=0, now_ts=100.0)

    assert processed is True
    assert client.state["mission"]["stop_reason"] is None
    assert client.state["last_action"]["throttle"] > 0.0
    assert client.sensor_rig.read_calls == [False]
    assert client.state["sensors"]["rear_preview"]["status"] == "disabled"
    assert client.state["sensors"]["rear"]["status"] == "disabled"


def test_empty_sensor_snapshot_is_json_safe_with_additive_fields():
    snapshot = hardware.empty_sensor_snapshot()
    snapshot["primary_rgb"]["frame_age_ms"] = 12.5
    snapshot["primary_rgb"]["used_for"] = [
        "lane_following",
        "apriltag",
        "forward_preview",
    ]
    snapshot["sidecar_depth_imu"]["depth_status"] = "available"
    snapshot["sidecar_depth_imu"]["depth_frame_age_ms"] = 3.25
    snapshot["sidecar_depth_imu"]["imu_status"] = "unavailable"
    snapshot["sidecar_depth_imu"]["imu_frame_age_ms"] = None
    snapshot["rear_preview"]["frame_age_ms"] = None
    snapshot["depth"] = {
        "role": "depth",
        "source": "realsense",
        "status": "available",
        "frame_age_ms": 3.25,
        "used_for": ["obstacle_stop"],
    }
    snapshot["imu"] = {
        "role": "imu",
        "source": "realsense",
        "status": "unavailable",
        "frame_age_ms": None,
        "used_for": ["state_context"],
    }
    snapshot["rear"] = {
        "role": "rear",
        "source": "cam1",
        "status": "disabled",
        "frame_age_ms": None,
        "used_for": ["rear_preview_only"],
    }

    encoded = json.dumps(snapshot)
    decoded = json.loads(encoded)

    assert decoded["primary_rgb"]["frame_age_ms"] == 12.5
    assert decoded["depth"]["used_for"] == ["obstacle_stop"]
    assert decoded["imu"]["status"] == "unavailable"
    assert decoded["rear"]["role"] == "rear"


def test_sidecar_alias_nodes_report_modality_specific_status_and_age():
    primary = StubCamera(frame=np.zeros((120, 160, 3), dtype=np.uint8))
    sidecar = StubCamera(
        depth=np.full((120, 160), 1000, dtype=np.uint16),
        imu=None,
    )
    rig = hardware.CompositeSensorRig(
        primary_rgb_camera=primary,
        primary_rgb_config={"role": "primary_rgb", "type": "csi", "enabled": True},
        sidecar_depth_imu_sensor=sidecar,
        sidecar_depth_imu_config={"role": "sidecar_depth_imu", "type": "realsense", "enabled": True},
    )

    packet = rig.read()
    snapshot = packet["sensor_snapshot"]

    assert snapshot["sidecar_depth_imu"]["depth_status"] == "available"
    assert snapshot["sidecar_depth_imu"]["imu_status"] == "unavailable"
    assert snapshot["depth"]["status"] == "available"
    assert snapshot["depth"]["frame_age_ms"] is not None
    assert snapshot["imu"]["status"] == "unavailable"
    assert snapshot["imu"]["frame_age_ms"] is None


def test_build_sensor_rig_keeps_sidecar_observable_when_primary_init_fails():
    configs = [
        {"role": "primary_rgb", "type": "csi", "sensor_id": 0, "enabled": True},
        {"role": "sidecar_depth_imu", "type": "realsense", "enabled": True},
    ]

    def fake_get_camera(config, enable_depth=True):
        if config.get("role") == "primary_rgb":
            raise RuntimeError("CAM0 init failed")
        return StubCamera(depth=np.full((10, 10), 1000, dtype=np.uint16), imu={})

    with patch.object(hardware, "get_camera", side_effect=fake_get_camera):
        rig = hardware.build_sensor_rig(configs, enable_depth=True)

    snapshot = rig.snapshot()

    assert rig.primary_rgb_camera is None
    assert rig.sidecar_depth_imu_sensor is not None
    assert snapshot["primary_rgb"]["status"] == "unavailable"
    assert snapshot["primary_rgb"]["error"] == "CAM0 init failed"
    assert snapshot["sidecar_depth_imu"]["status"] == "configured"
    assert snapshot["depth"]["status"] == "configured"
    assert snapshot["imu"]["status"] == "configured"

import sys
from unittest.mock import MagicMock

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

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

# Mock smbus2 to avoid fcntl import errors on Windows
sys.modules['smbus2'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.models'] = MagicMock()

from fastapi.testclient import TestClient
from main import app, ClientConfig, _drain_log_buffer, _current_password
import logging

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_unauthorized_access():
    # Without X-Api-Key
    response = client.get("/status")
    assert response.status_code == 401
    
    # With wrong X-Api-Key
    response = client.get("/status", headers={"X-Api-Key": "wrong"})
    assert response.status_code == 401

def test_authorized_access():
    response = client.get("/status", headers={"X-Api-Key": _current_password})
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "paused" in data

def test_log_buffer():
    # clear initial buffer
    _drain_log_buffer()
    
    # create logs
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    logger.info("Test log message 1")
    logger.error("Test error message 2")
    
    logs = _drain_log_buffer()
    assert len(logs) == 2
    assert "Test log message 1" in logs[0]["message"]
    assert logs[0]["level"] == "INFO"
    assert "Test error message 2" in logs[1]["message"]
    assert logs[1]["level"] == "ERROR"

    # verify empty after drain
    assert len(_drain_log_buffer()) == 0


def test_legacy_camera_config_still_parses():
    payload = {
        "device": "cuda",
        "architecture": "resnet18",
        "cameras": [
            {
                "type": "opencv",
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        ],
        "control_model_type": "pytorch",
        "control_model": "best_model.pth",
        "throttle_mode": "fixed",
        "fixed_throttle_value": 0.22,
        "action_loop": ["control", "api"],
        "ip": "0.0.0.0",
        "port": 8000,
        "password": _current_password,
    }

    config = ClientConfig(**payload)

    assert config.password == _current_password
    assert config.cameras[0].type == "opencv"
    assert config.cameras[0].index == 0
    assert config.cameras[0].width == 640
    assert config.cameras[0].height == 480
    assert config.cameras[0].fps == 30


def test_role_based_camera_config_parses():
    payload = {
        "device": "cuda",
        "architecture": "resnet101",
        "preprocess_profile": "cam0_fisheye_v1",
        "cameras": [
            {
                "role": "primary_rgb",
                "type": "csi",
                "sensor_id": 0,
                "width": 640,
                "height": 480,
                "fps": 15,
                "flip_method": 2,
                "enabled": True,
            },
            {
                "role": "sidecar_depth_imu",
                "type": "realsense",
                "width": 640,
                "height": 480,
                "fps": 15,
                "enabled": True,
            },
            {
                "role": "rear_preview",
                "type": "csi",
                "sensor_id": 1,
                "width": 640,
                "height": 480,
                "fps": 15,
                "flip_method": 2,
                "enabled": False,
            },
        ],
        "control_model_type": "tensorrt",
        "control_model": "best_model.pth",
        "throttle_mode": "fixed",
        "fixed_throttle_value": 0.22,
        "action_loop": ["control", "api"],
        "ip": "0.0.0.0",
        "port": 8000,
        "password": _current_password,
        "mission": {
            "enabled": True,
            "route_name": "expo_route",
            "tag_ids": {"start_home": 10, "checkpoint": 20, "goal": 30},
            "depth_stop": {"enabled": True, "threshold_m": 0.6},
        },
    }

    config = ClientConfig(**payload)

    assert config.mission["enabled"] is True
    assert config.mission["route_name"] == "expo_route"
    assert config.mission["tag_ids"]["checkpoint"] == 20
    assert len(config.cameras) == 3

    primary = config.cameras[0]
    sidecar = config.cameras[1]
    rear = config.cameras[2]

    if hasattr(config, "preprocess_profile"):
        assert config.preprocess_profile == "cam0_fisheye_v1"

    if hasattr(primary, "role"):
        assert primary.role == "primary_rgb"
        assert primary.sensor_id == 0
        assert primary.flip_method == 2
        assert primary.enabled is True

    if hasattr(sidecar, "role"):
        assert sidecar.role == "sidecar_depth_imu"
        assert sidecar.enabled is True

    if hasattr(rear, "role"):
        assert rear.role == "rear_preview"
        assert rear.enabled is False


def test_example_cam0_primary_config_parses_if_present():
    example_path = Path(__file__).resolve().parent.parent / "docs" / "examples" / "client_config_cam0_primary.json"
    assert example_path.exists(), f"Missing required example config: {example_path}"

    payload = json.loads(example_path.read_text())
    config = ClientConfig(**payload)

    assert config.cameras
    assert config.cameras[0].role == "primary_rgb"
    assert config.cameras[0].type == "csi"
    assert config.cameras[0].sensor_id == 0

    if hasattr(config, "preprocess_profile"):
        assert config.preprocess_profile == "cam0_fisheye_v1"
    assert config.password == "changeme"

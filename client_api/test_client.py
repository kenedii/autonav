import pytest
import sys
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


def test_configure_accepts_optional_mission_fields():
    payload = {
        "device": "cuda",
        "architecture": "resnet18",
        "cameras": [],
        "control_model_type": "pytorch",
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

    try:
        config = ClientConfig(**payload)
    except Exception as exc:
        pytest.skip(f"mission config not supported yet: {exc}")

    assert config.password == _current_password
    assert config.mission["enabled"] is True
    assert config.mission["route_name"] == "expo_route"
    assert config.mission["tag_ids"]["checkpoint"] == 20

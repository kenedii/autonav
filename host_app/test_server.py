import pytest
from fastapi.testclient import TestClient
from server import app, db, _safe_car

client = TestClient(app)

def test_add_test_client():
    response = client.post("/api/test-client")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "added"
    assert "id" in data

    # Check if logs were seeded
    client_id = data["id"]
    car_resp = client.get("/api/cars")
    assert car_resp.status_code == 200
    car = next(item for item in car_resp.json() if item["id"] == client_id)
    sensors = car["details"]["state"]["sensors"]
    assert sensors["forward_preview_role"] == "primary_rgb"
    assert sensors["primary_rgb"]["role"] == "primary_rgb"
    assert sensors["sidecar_depth_imu"]["role"] == "sidecar_depth_imu"
    assert sensors["rear_preview"]["role"] == "rear_preview"
    assert car["details"]["config"]["cameras"][0]["role"] == "primary_rgb"
    assert car["details"]["config"]["preprocess_profile"] == "cam0_fisheye_v1"
    assert car["details"]["state"]["location"]["imu"]["accel"] == [0.01, -0.02, 0.98]
    assert car["details"]["state"]["specs"]["cameras"][1]["role"] == "sidecar_depth_imu"

    logs_resp = client.get(f"/api/cars/{client_id}/logs")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert len(logs_data["logs"]) > 0

def test_add_car():
    response = client.post("/api/cars", json={
        "name": "TestCar1",
        "ip": "192.168.1.100",
        "port": 8000,
        "password": "pass"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "added"
    assert "car" in data
    assert data["car"]["name"] == "TestCar1"
    assert "password" not in data["car"]

def test_get_cars():
    response = client.get("/api/cars")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    for car in data:
        assert "password" not in car


def test_safe_car_redacts_passwords():
    record = {
        "id": "car-1",
        "name": "Car 1",
        "password": "top-secret",
        "fernet": object(),
        "details": {
            "config": {
                "password": "nested-secret",
                "route_name": "expo_route",
                "cameras": [{"role": "primary_rgb", "type": "csi"}],
            },
            "state": {
                "sensors": {
                    "forward_preview_role": "primary_rgb",
                    "primary_rgb": {"role": "primary_rgb", "status": "available"},
                }
            }
        },
    }

    safe = _safe_car(record)

    assert "fernet" not in safe
    assert "password" not in safe
    assert safe["details"]["config"]["route_name"] == "expo_route"
    assert "password" not in safe["details"]["config"]
    assert safe["details"]["state"]["sensors"]["forward_preview_role"] == "primary_rgb"
    assert safe["details"]["state"]["sensors"]["primary_rgb"]["role"] == "primary_rgb"


def test_car_log_since_filters_incrementally():
    client_id = client.post("/api/test-client").json()["id"]
    db.logs[client_id].clear()
    db.logs[client_id].append({"timestamp": 100.0, "level": "INFO", "message": "old"})
    db.logs[client_id].append({"timestamp": 200.0, "level": "INFO", "message": "new"})

    logs_resp = client.get(f"/api/cars/{client_id}/logs?since=150.0")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert [entry["message"] for entry in logs_data["logs"]] == ["new"]


def test_delete_car():
    response = client.post("/api/cars", json={
        "name": "DeleteMe",
        "ip": "192.168.1.111",
        "port": 8000,
        "password": "pass"
    })
    car_id = response.json()["car"]["id"]

    del_res = client.delete(f"/api/cars/{car_id}")
    assert del_res.status_code == 200

    missing = client.get("/api/cars")
    assert all(car["id"] != car_id for car in missing.json())

def test_delete_car_logs():
    # first setup a client
    res = client.post("/api/test-client")
    client_id = res.json()["id"]
    
    # delete logs
    del_res = client.delete(f"/api/cars/{client_id}/logs")
    assert del_res.status_code == 200
    
    # verify empty
    logs_resp = client.get(f"/api/cars/{client_id}/logs")
    assert logs_resp.json()["logs"] == []

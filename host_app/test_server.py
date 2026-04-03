import pytest
from fastapi.testclient import TestClient
from server import app, db

client = TestClient(app)

def test_add_test_client():
    response = client.post("/api/test-client")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "added"
    assert "id" in data
    
    # Check if logs were seeded
    client_id = data["id"]
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

def test_get_cars():
    response = client.get("/api/cars")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

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

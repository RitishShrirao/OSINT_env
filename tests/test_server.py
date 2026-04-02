from fastapi.testclient import TestClient

from server import app


client = TestClient(app)


def test_server_health():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_server_environment_metadata():
    response = client.get("/api/environment")
    assert response.status_code == 200
    body = response.json()
    assert "action_space" in body
    assert "observation_space" in body
    assert "summary" in body


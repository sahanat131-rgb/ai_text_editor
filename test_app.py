from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_is_up():
    r = client.get("/")
    assert r.status_code == 200


def test_predict_returns_valid_response():
    r = client.post("/predict", json={"text": "my flight was delayed again, so annoying"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("AI-generated", "Human-written")
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_without_text_is_rejected():
    r = client.post("/predict", json={})
    assert r.status_code == 422
"""Backend endpoints and survey scoring (read-only – does not change saved data)."""
from fastapi.testclient import TestClient

import app as backend

client = TestClient(backend.app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_sus_scoring_matches_standard():
    assert backend.sus_score([5, 1] * 5) == 100.0       # best answers
    assert backend.sus_score([1, 5] * 5) == 0.0         # worst answers
    assert backend.sus_score([3] * 10) == 50.0          # neutral
    assert backend.sus_grade(86) == "Excellent" and backend.sus_grade(70) == "OK"


def test_invalid_eye_mode_rejected():
    r = client.put("/api/config", json={"eye_mode": "magic"})
    assert r.status_code == 400


def test_fleet_summary_shape():
    f = client.get("/api/fleet").json()
    assert {"drivers", "totals", "duty_limit_hours"} <= set(f)


def test_overpass_proxy_rejects_other_queries():
    assert client.get("/api/overpass", params={"q": "DROP TABLE"}).status_code == 400

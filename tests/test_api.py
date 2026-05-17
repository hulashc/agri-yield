"""
Integration tests for FastAPI endpoints.
Uses a toy XGBoost model, mock weather/drift, and 3 test fields.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_returns_200(app):
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["fields_loaded"] is True
    assert body["build_version"] == "dev"


def test_health_reports_not_loaded_when_model_missing(monkeypatch):
    monkeypatch.setenv("PICKLE_MODEL_PATH", "/nonexistent/model.pkl")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "disabled")
    monkeypatch.setenv("REDIS_URL", "")
    with (
        patch("ingestion.openmeteo_live.get_live_features") as mock_wx,
        patch("monitoring.psi_detector.evaluate_drift") as mock_drift,
    ):
        mock_wx.return_value = {"stale_features": True}
        mock_drift.return_value = {"drift_warning": False, "drift_level": "none", "max_psi": 0.0, "psi_scores": {}}
        # Reload serving.model to reset _model + re-read PICKLE_MODEL_PATH from env
        import importlib

        import serving.model
        importlib.reload(serving.model)
        import serving.app
        importlib.reload(serving.app)
        from serving.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is False


def test_fields_returns_all_fields(app):
    client = TestClient(app)
    resp = client.get("/fields")
    assert resp.status_code == 200
    body = resp.json()
    assert "fields" in body
    assert len(body["fields"]) == 3
    assert body["model_version"] == "pkl-ci"


def test_fields_contains_expected_keys(app):
    client = TestClient(app)
    resp = client.get("/fields")
    fields = resp.json()["fields"]
    field = fields[0]
    assert "field_id" in field
    assert "predicted_yield_kg_ha" in field
    assert "lower_bound" in field
    assert "upper_bound" in field
    assert "drift_warning" in field
    assert "temp_c" in field
    assert "rainfall_mm" in field
    assert "solar_rad_mj_m2" in field


def test_fields_yields_are_positive_floats(app):
    client = TestClient(app)
    fields = client.get("/fields").json()["fields"]
    for f in fields:
        assert f["predicted_yield_kg_ha"] is not None
        assert f["predicted_yield_kg_ha"] > 0
        assert f["lower_bound"] < f["predicted_yield_kg_ha"] < f["upper_bound"]


def test_fields_different_per_field(app):
    client = TestClient(app)
    fields = client.get("/fields").json()["fields"]
    yields = [f["predicted_yield_kg_ha"] for f in fields]
    # At least 2 different values across 3 fields with different lat/lon
    assert len(set(yields)) >= 2


def test_fields_no_errors(app):
    client = TestClient(app)
    fields = client.get("/fields").json()["fields"]
    errors = [f.get("error") for f in fields if f.get("error")]
    assert errors == [], f"Unexpected errors: {errors}"


def test_predict_single_field(app):
    client = TestClient(app)
    resp = client.post("/predict", json={"field_id": "F001", "event_timestamp": "2026-05-17T00:00:00"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["field_id"] == "F001"
    assert body["predicted_yield_kg_ha"] > 0
    assert body["lower_bound"] < body["upper_bound"]
    assert body["model_version"] == "pkl-ci"


def test_predict_unknown_field_returns_404(app):
    client = TestClient(app)
    resp = client.post("/predict", json={"field_id": "NONEXISTENT", "event_timestamp": "2026-05-17T00:00:00"})
    assert resp.status_code == 404


def test_predict_missing_body_returns_422(app):
    client = TestClient(app)
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_metrics_endpoint_returns_prometheus(app):
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "version=" in resp.headers["content-type"]
    text = resp.text
    assert "agri_yield_prediction_latency_seconds" in text
    assert "agri_yield_predictions_total" in text


def test_map_ui_returns_html(app):
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Agri Yield" in resp.text
    assert "leaflet" in resp.text.lower()

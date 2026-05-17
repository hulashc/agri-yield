import importlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"


def _train_toy_model() -> bytes:
    rng = np.random.default_rng(42)
    n = 500
    rows = []
    for i in range(n):
        rows.append({col: float(rng.random()) for col in FEATURE_COLS})
    df = pd.DataFrame(rows)
    df[TARGET] = df["lat"] * 2000 + df["precipitation_sum"] * 300 + rng.normal(0, 200, n)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(df[FEATURE_COLS], df[TARGET])
    return pickle.dumps(model)


@pytest.fixture(scope="session")
def toy_model_bytes() -> bytes:
    return _train_toy_model()


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "disabled")
    monkeypatch.setenv("REDIS_URL", "")

    small_fields = pd.DataFrame({
        "field_id": ["F001", "F002", "F003"],
        "name": ["Alpha Farm", "Bravo Farm", "Charlie Farm"],
        "lat": [52.0, 53.0, 51.5],
        "lon": [-1.5, -2.0, 0.5],
        "area_ha": [50.0, 80.0, 30.0],
        "crop_type": ["winter_wheat", "winter_barley", "oilseed_rape"],
        "region": ["East Anglia", "Yorkshire", "Midlands"],
        "soil_type": ["clay", "sandy_loam", "loam"],
    })
    csv_path = tmp_path / "uk_fields.csv"
    small_fields.to_csv(csv_path, index=False)
    monkeypatch.setenv("FIELDS_CSV_PATH", str(csv_path))


@pytest.fixture
def model_pkl_path(monkeypatch, toy_model_bytes, tmp_path) -> str:
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(toy_model_bytes)
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(pkl))
    return str(pkl)


@pytest.fixture
def app(model_pkl_path):
    """Build a FastAPI test app with mocked weather and drift.

    Patches are applied BEFORE importing serving.app so the import
    binds mock objects instead of real implementations.
    """
    with (
        patch("ingestion.openmeteo_live.get_live_features") as mock_wx,
        patch("monitoring.psi_detector.evaluate_drift") as mock_drift,
    ):
        mock_wx.return_value = {
            "t2m_max_today": 15.0,
            "rainfall_today_mm": 2.0,
            "solar_radiation_today": 12.0,
            "t2m_min_today": 7.0,
            "et0_today": 3.0,
            "stale_features": False,
            "forecast_date": "2026-05-17",
        }
        mock_drift.return_value = {
            "drift_warning": False,
            "drift_level": "none",
            "max_psi": 0.05,
            "psi_scores": {},
        }
        import serving.model
        importlib.reload(serving.model)
        from serving.app import app, _startup_load
        import asyncio
        asyncio.run(_startup_load())
        yield app

import importlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"


def _make_bundle(rng=None) -> dict:
    """Build a real 3-model bundle (mean + lower + upper) for tests."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = 500
    rows = [{col: float(rng.random()) for col in FEATURE_COLS} for _ in range(n)]
    df = pd.DataFrame(rows)
    df[TARGET] = df["lat"] * 2000 + df["precipitation_sum"] * 300 + rng.normal(0, 200, n)
    base = dict(n_estimators=50, max_depth=4, random_state=42)
    mean_m = xgb.XGBRegressor(**base)
    mean_m.fit(df[FEATURE_COLS], df[TARGET])
    lower_m = xgb.XGBRegressor(**base, objective="reg:quantileerror", quantile_alpha=0.10)
    lower_m.fit(df[FEATURE_COLS], df[TARGET])
    upper_m = xgb.XGBRegressor(**base, objective="reg:quantileerror", quantile_alpha=0.90)
    upper_m.fit(df[FEATURE_COLS], df[TARGET])
    importance = {
        feat: float(score)
        for feat, score in zip(FEATURE_COLS, mean_m.feature_importances_, strict=False)
    }
    return {
        "mean": mean_m,
        "lower": lower_m,
        "upper": upper_m,
        "rmse": 450.0,
        "feature_importance": importance,
        "trained_at": "2026-06-11T00:00:00+00:00",
        "dataset_source": "synthetic",
        "feature_cols": FEATURE_COLS,
        "quantile_lower": 0.10,
        "quantile_upper": 0.90,
        "ci_coverage": "80%",
        "ordering_violations": 0,
    }


@pytest.fixture(scope="session")
def toy_bundle() -> dict:
    """Session-scoped 3-model bundle — built once, reused across all tests."""
    return _make_bundle()


@pytest.fixture(scope="session")
def toy_model_bytes(toy_bundle) -> bytes:
    """Legacy bytes fixture kept for backward compat with test_splits etc."""
    return pickle.dumps(toy_bundle["mean"])


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "disabled")
    monkeypatch.setenv("REDIS_URL", "")  # disable Redis in tests — use in-memory fallback

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
def model_bundle_path(monkeypatch, toy_bundle, tmp_path) -> str:
    """Write bundle to tmp_path and point env vars at it."""
    bundle_path = tmp_path / "model_bundle.pkl"
    bundle_path.write_bytes(pickle.dumps(toy_bundle))
    monkeypatch.setenv("PICKLE_BUNDLE_PATH", str(bundle_path))
    # Also write legacy pkl for any code that still reads model.pkl
    pkl_path = tmp_path / "model.pkl"
    pkl_path.write_bytes(pickle.dumps(toy_bundle["mean"]))
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(pkl_path))
    return str(bundle_path)


@pytest.fixture
def model_pkl_path(model_bundle_path) -> str:
    """Alias kept for tests that depend on the old fixture name."""
    return model_bundle_path


@pytest.fixture
def app(model_bundle_path):
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
        import asyncio

        from serving.app import _startup_load, app
        asyncio.run(_startup_load())
        yield app

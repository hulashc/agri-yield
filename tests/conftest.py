import importlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"


def _make_toy_bundle(rng: np.random.Generator, n: int = 500) -> dict:
    """Train a minimal mean + lower + upper bundle for use in tests."""
    rows = [{col: float(rng.random()) for col in FEATURE_COLS} for _ in range(n)]
    df = pd.DataFrame(rows)
    y = df["lat"] * 2000 + df["precipitation_sum"] * 300 + rng.normal(0, 200, n)

    base_params = dict(n_estimators=50, max_depth=4, random_state=42)

    mean_m = xgb.XGBRegressor(**base_params)
    mean_m.fit(df[FEATURE_COLS], y)

    lower_m = xgb.XGBRegressor(
        **base_params, objective="reg:quantileerror", quantile_alpha=0.10
    )
    lower_m.fit(df[FEATURE_COLS], y)

    upper_m = xgb.XGBRegressor(
        **base_params, objective="reg:quantileerror", quantile_alpha=0.90
    )
    upper_m.fit(df[FEATURE_COLS], y)

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
        "dataset_source": "synthetic_test",
        "feature_cols": FEATURE_COLS,
        "quantile_lower": 0.10,
        "quantile_upper": 0.90,
        "ci_coverage": "80%",
        "ordering_violations": 0,
    }


@pytest.fixture(scope="session")
def toy_bundle() -> dict:
    rng = np.random.default_rng(42)
    return _make_toy_bundle(rng)


@pytest.fixture(scope="session")
def toy_model_bytes(toy_bundle) -> bytes:
    """Legacy bytes fixture for tests that still load model.pkl directly."""
    return pickle.dumps(toy_bundle["mean"])


@pytest.fixture(autouse=True)
def env_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "disabled")
    monkeypatch.setenv("REDIS_URL", "")  # force in-memory buffer in all tests

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
    """Write the full model bundle to a temp file and point env vars at it."""
    bundle_path = tmp_path / "model_bundle.pkl"
    bundle_path.write_bytes(pickle.dumps(toy_bundle))
    monkeypatch.setenv("PICKLE_BUNDLE_PATH", str(bundle_path))

    # Also write legacy pkl for backward-compat code paths
    pkl_path = tmp_path / "model.pkl"
    pkl_path.write_bytes(pickle.dumps(toy_bundle["mean"]))
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(pkl_path))

    return str(bundle_path)


@pytest.fixture
def model_pkl_path(monkeypatch, toy_model_bytes, tmp_path) -> str:
    """Legacy fixture: single model.pkl only (no quantile models)."""
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(toy_model_bytes)
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(pkl))
    # Clear bundle path so loader falls through to legacy pkl
    monkeypatch.setenv("PICKLE_BUNDLE_PATH", str(tmp_path / "nonexistent_bundle.pkl"))
    return str(pkl)


@pytest.fixture
def app(model_bundle_path):
    """
    Build a FastAPI test app with mocked weather and drift.

    Uses the full model bundle (mean + lower + upper) so CI endpoints
    return real quantile-derived bounds, not the ±15% heuristic fallback.
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
            "drift_level": "green",
            "max_psi": 0.05,
            "psi_scores": {},
        }
        import serving.model
        importlib.reload(serving.model)
        import asyncio
        from serving.app import _startup_load, app
        asyncio.run(_startup_load())
        yield app

import importlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from training.model_bundle import ModelBundle
from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"


def _make_toy_df(n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """Minimal synthetic dataset with all FEATURE_COLS."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({col: rng.random(n) for col in FEATURE_COLS})
    y = df["lat"] * 2000 + df["precipitation_sum"] * 300 + rng.normal(0, 200, n)
    return df, pd.Series(y)


def _train_toy_bundle() -> ModelBundle:
    """Train a tiny 3-model bundle for test fixtures."""
    X, y = _make_toy_df(500)
    base_params = dict(n_estimators=30, max_depth=3, random_state=42)

    mean_model = xgb.XGBRegressor(**base_params)
    mean_model.fit(X, y)

    lower_model = xgb.XGBRegressor(
        **base_params, objective="reg:quantileerror", quantile_alpha=0.10
    )
    lower_model.fit(X, y)

    upper_model = xgb.XGBRegressor(
        **base_params, objective="reg:quantileerror", quantile_alpha=0.90
    )
    upper_model.fit(X, y)

    feature_importance = {
        feat: float(v)
        for feat, v in zip(
            FEATURE_COLS,
            mean_model.feature_importances_,
        )
    }

    return ModelBundle(
        mean_model=mean_model,
        lower_model=lower_model,
        upper_model=upper_model,
        feature_importance=feature_importance,
        training_meta={
            "trained_at": "2026-06-11T00:00:00+00:00",
            "rmse_kg_ha": 250.0,
            "mae_kg_ha": 180.0,
            "pi_coverage_pct": 81.5,
            "interval_label": "80% prediction interval",
            "split_policy": "temporal — no future leakage",
            "confidence_method": "xgboost_quantile_regression",
            "n_train": 400,
            "n_test": 100,
            "lower_quantile": 0.10,
            "upper_quantile": 0.90,
        },
    )


def _train_toy_model() -> bytes:
    """Legacy: bare XGBRegressor pickle bytes (for backward-compat tests)."""
    X, y = _make_toy_df(500)
    model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)
    return pickle.dumps(model)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def toy_bundle() -> ModelBundle:
    return _train_toy_bundle()


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
def bundle_pkl_path(monkeypatch, toy_bundle, tmp_path) -> str:
    """Write toy bundle to tmp dir and point BUNDLE_MODEL_PATH at it."""
    pkl = tmp_path / "model_bundle.pkl"
    toy_bundle.save(pkl)
    monkeypatch.setenv("BUNDLE_MODEL_PATH", str(pkl))
    # Make sure legacy path points at a nonexistent file
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(tmp_path / "nonexistent.pkl"))
    return str(pkl)


@pytest.fixture
def model_pkl_path(monkeypatch, toy_model_bytes, tmp_path) -> str:
    """Legacy: write bare model.pkl and point PICKLE_MODEL_PATH at it."""
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(toy_model_bytes)
    monkeypatch.setenv("PICKLE_MODEL_PATH", str(pkl))
    # Ensure bundle path is absent so legacy loader is exercised
    monkeypatch.setenv("BUNDLE_MODEL_PATH", str(tmp_path / "nonexistent_bundle.pkl"))
    return str(pkl)


@pytest.fixture
def app(bundle_pkl_path):
    """
    FastAPI test app using the toy ModelBundle.
    Patches weather + drift before importing serving.app.
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

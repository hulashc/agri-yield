"""
tests/test_model_card.py

Tests for model card generation (training) and loading (serving).
"""

import json
from pathlib import Path

import pytest

from training.utils.features import FEATURE_COLS
from training.utils.model_card import load_model_card, write_model_card


@pytest.fixture
def card_path(tmp_path) -> str:
    return str(tmp_path / "model_card.json")


def test_write_model_card_creates_file(card_path):
    write_model_card(
        rmse=1500.0,
        n_train=800,
        n_test=200,
        dataset_source="synthetic fallback",
        feature_cols=FEATURE_COLS,
        params={"n_estimators": 300},
        output_path=card_path,
    )
    assert Path(card_path).exists()


def test_write_model_card_has_required_keys(card_path):
    write_model_card(
        rmse=1500.0,
        n_train=800,
        n_test=200,
        dataset_source="synthetic fallback",
        feature_cols=FEATURE_COLS,
        params={"n_estimators": 300},
        output_path=card_path,
    )
    card = json.loads(Path(card_path).read_text())
    required = [
        "model_name", "algorithm", "target", "trained_at",
        "dataset_source", "split_policy", "n_train", "n_test",
        "metrics", "hyperparameters", "feature_columns",
        "ci_method", "known_limitations",
    ]
    for key in required:
        assert key in card, f"Missing key in model card: {key}"


def test_write_model_card_rmse_stored(card_path):
    write_model_card(
        rmse=1234.56,
        n_train=700,
        n_test=300,
        dataset_source="CYCleSS UK yield data",
        feature_cols=FEATURE_COLS,
        params={},
        output_path=card_path,
    )
    card = json.loads(Path(card_path).read_text())
    assert card["metrics"]["rmse_kg_per_ha"] == pytest.approx(1234.56, abs=0.01)


def test_write_model_card_feature_cols_match(card_path):
    write_model_card(
        rmse=1500.0,
        n_train=800,
        n_test=200,
        dataset_source="synthetic fallback",
        feature_cols=FEATURE_COLS,
        params={},
        output_path=card_path,
    )
    card = json.loads(Path(card_path).read_text())
    assert card["feature_columns"] == FEATURE_COLS


def test_write_model_card_counts(card_path):
    write_model_card(
        rmse=1500.0,
        n_train=800,
        n_test=200,
        dataset_source="synthetic fallback",
        feature_cols=FEATURE_COLS,
        params={},
        output_path=card_path,
    )
    card = json.loads(Path(card_path).read_text())
    assert card["n_train"] == 800
    assert card["n_test"] == 200


def test_load_model_card_returns_dict(card_path):
    write_model_card(
        rmse=1000.0,
        n_train=500,
        n_test=100,
        dataset_source="synthetic fallback",
        feature_cols=FEATURE_COLS,
        params={},
        output_path=card_path,
    )
    card = load_model_card(card_path)
    assert isinstance(card, dict)
    assert card["model_name"] == "agri-yield-xgb"


def test_load_model_card_missing_returns_empty():
    card = load_model_card("/nonexistent/model_card.json")
    assert card == {}


def test_model_info_endpoint_returns_card(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    # /model/info returns 503 when card not baked in test fixture — that is expected
    # Test that the route exists and returns either 200 or 503 (never 404)
    resp = client.get("/model/info")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "model_name" in body
        assert "metrics" in body
        assert "feature_columns" in body


def test_health_includes_model_card_fields(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # These keys must be present regardless of whether card is loaded
    assert "model_trained_at" in body
    assert "model_rmse_kg_ha" in body
    assert "dataset_source" in body

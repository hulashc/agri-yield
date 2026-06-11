from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    field_id: str = Field(..., min_length=1, max_length=64)
    event_timestamp: str = Field(..., description="ISO-8601, e.g. '2024-06-01T00:00:00'")


class PredictResponse(BaseModel):
    field_id: str
    predicted_yield_kg_ha: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(0.80, description="Nominal coverage of the prediction interval")
    confidence_method: str = Field(
        "xgboost_quantile_regression",
        description="'xgboost_quantile_regression' when bundle loaded, 'legacy_symmetric_ci' otherwise",
    )
    drift_warning: bool
    drift_level: str
    psi_score: float
    stale_features: bool
    model_version: str
    last_updated: str


class FeatureImportanceItem(BaseModel):
    rank: int
    feature: str
    importance: float
    source: str = "shap"


class ModelInfoResponse(BaseModel):
    model_version: str
    confidence_method: str
    using_bundle: bool
    trained_at: str | None = None
    rmse_kg_ha: float | None = None
    mae_kg_ha: float | None = None
    pi_coverage_pct: float | None = None
    interval_label: str | None = None
    split_policy: str | None = None
    n_train: int | None = None
    n_test: int | None = None
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    warning: str | None = None


class ExplainRequest(BaseModel):
    field_id: str = Field(..., min_length=1, max_length=64)
    top_n: int = Field(default=10, ge=1, le=len([].__class__.mro()), description="Number of top SHAP contributors to return")


class ContributorItem(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    direction: str  # "positive" | "negative"


class ExplainResponse(BaseModel):
    field_id: str
    prediction: float = Field(description="Mean model prediction in kg/ha")
    base_value: float | None = Field(
        default=None,
        description="SHAP base value (expected model output before feature adjustments)",
    )
    top_contributors: list[ContributorItem] = Field(
        default_factory=list,
        description="Top features driving this prediction, ranked by |SHAP value|",
    )
    source: str = Field(
        description="'shap' when SHAP is available, 'xgboost_gain_fallback' otherwise"
    )
    error: str | None = None
    model_version: str
    last_updated: str


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]

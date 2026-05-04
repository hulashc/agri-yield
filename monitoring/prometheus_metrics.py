from prometheus_client import Counter, Histogram, Gauge, Summary

PREDICTION_LATENCY = Histogram(
    "agri_yield_prediction_latency_seconds",
    "End-to-end latency for /predict endpoint",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTIONS_TOTAL = Counter(
    "agri_yield_predictions_total",
    "Total prediction requests",
    labelnames=["crop_type", "region"],
)
PREDICTION_YIELD_KG_HA = Summary(
    "agri_yield_predicted_yield_kg_ha",
    "Distribution of predicted yield values",
    labelnames=["crop_type"],
)
PREDICTION_CI_WIDTH = Histogram(
    "agri_yield_confidence_interval_width",
    "Width of 80% CI in kg/ha",
    buckets=[50, 100, 200, 400, 800, 1600],
)
DRIFT_WARNINGS_TOTAL = Counter(
    "agri_yield_drift_warnings_total",
    "PSI drift warnings fired",
    labelnames=["field_id", "feature_name"],
)
PSI_SCORE = Gauge(
    "agri_yield_psi_score",
    "Latest PSI score per feature",
    labelnames=["feature_name"],
)
DRIFT_LEVEL = Gauge(
    "agri_yield_drift_level",
    "Drift level: 0=green 1=amber 2=red",
    labelnames=["field_id"],
)
MODEL_VERSION = Gauge(
    "agri_yield_model_version",
    "Currently deployed model version",
)
STALE_FEATURE_REQUESTS = Counter(
    "agri_yield_stale_feature_requests_total",
    "Requests served with stale features",
    labelnames=["field_id"],
)
RETRAIN_EVENTS = Counter(
    "agri_yield_retrain_events_total",
    "Retraining pipeline executions",
    labelnames=["trigger_reason"],
)
RETRAIN_PROMOTED = Counter(
    "agri_yield_retrain_promoted_total",
    "Retrained models promoted to Production",
)
LAST_RETRAIN_TIMESTAMP = Gauge(
    "agri_yield_last_retrain_unix_timestamp",
    "Unix timestamp of last retraining run",
)

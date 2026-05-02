@'
# Model Card — [Model Name]

**Version:** <!-- e.g. v1.0 -->  
**Registered in MLflow:** <!-- run ID -->  
**DVC Dataset Ref:** <!-- commit hash of weekly_field_features.dvc -->  
**Date:** <!-- YYYY-MM-DD -->

---

## Intended Use

- **Primary use:** Weekly crop yield prediction per field
- **Intended users:** Agronomists, farm management systems
- **Out-of-scope uses:** Real-time per-minute decisions, non-agricultural domains

---

## Training Data

| Property | Value |
|---|---|
| Source | Feast offline store — `soil_features`, `weather_features`, `vegetation_features` |
| DVC ref | <!-- paste commit hash --> |
| Date range | <!-- e.g. 2023-01-01 to 2025-12-31 --> |
| Fields | <!-- number of unique field_ids --> |
| Seasons covered | Spring, Summer, Autumn, Winter |
| Crop types | <!-- list crop types in training set --> |

**Notes on imputed features:**  
<!-- Document what % of NDVI values were interpolated or proxied. -->

---

## Evaluation Metrics

### Overall

| Metric | Value |
|---|---|
| RMSE | <!-- --> |
| MAE | <!-- --> |
| R² | <!-- --> |

### By Season

| Season | RMSE | MAE | R² |
|---|---|---|---|
| Spring | | | |
| Summer | | | |
| Autumn | | | |
| Winter | | | |

### By Crop Type

| Crop | RMSE | MAE | R² |
|---|---|---|---|
| <!-- crop --> | | | |

---

## Known Limitations

- NDVI values flagged `ndvi_interpolated=True` or `ndvi_proxied=True` carry higher uncertainty
- NPK sensor data is sparse for certain regions — model performance may degrade for those fields
- Temporal train/test split used (no shuffle) — performance on never-seen crop types is not validated
- Model trained on synthetic sensor data — real-world calibration drift patterns may differ

---

## Drift Detection Thresholds

| Trigger | Threshold | Action |
|---|---|---|
| PSI > 0.2 on >30% of features | Data drift | Trigger retraining DAG |
| Season-matched RMSE degrades >10% | Concept drift | Trigger retraining DAG |
| Seasonal PSI exceedance (expected) | Not a trigger | Update reference window only |

---

## Lineage

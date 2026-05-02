import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a set of predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_by_crop(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    crop_col: str = "crop_type",
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down per crop type."""
    results = {}
    for crop, group in df.groupby(crop_col):
        results[str(crop)] = compute_metrics(
            group[y_true_col].values,
            group[y_pred_col].values,
        )
    return results

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: "np.ndarray[np.Any, np.Any]",
    y_pred: "np.ndarray[np.Any, np.Any]",
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for a set of predictions.

    Accepts any array-like (np.ndarray, pd.Series, ExtensionArray) and
    converts to a plain ndarray so mypy and sklearn are both satisfied.
    """
    y_true_arr: np.ndarray = np.asarray(y_true, dtype=float)
    y_pred_arr: np.ndarray = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_metrics_by_crop(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    crop_col: str = "crop_type",
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down per crop type."""
    results: dict[str, dict[str, float]] = {}
    for crop, group in df.groupby(crop_col):
        results[str(crop)] = compute_metrics(
            group[y_true_col].to_numpy(dtype=float),
            group[y_pred_col].to_numpy(dtype=float),
        )
    return results

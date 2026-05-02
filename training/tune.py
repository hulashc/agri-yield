"""
Optuna hyperparameter study for XGBoost.
Uses TPE sampler and TimeSeriesSplit cross-validation.
Never shuffles data.
"""

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from training.utils.metrics import compute_metrics
from training.utils.splits import temporal_train_test_split

TARGET = "yield_kg_per_ha"
NON_FEATURE_COLS = [TARGET, "week_start", "field_id"]


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Optuna objective function. Returns mean CV RMSE (lower is better).
    TPE sampler explores the hyperparameter space intelligently.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train[train_idx],
            y_train[train_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        preds = model.predict(X_train[val_idx])
        metrics = compute_metrics(y_train[val_idx], preds)
        rmse_scores.append(metrics["rmse"])

    return float(np.mean(rmse_scores))


def run_tuning(
    dataset_path: str = "data/features/weekly_field_features",
    n_trials: int = 50,
):
    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])

    for col in ["ndvi_interpolated", "ndvi_proxied"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    train_df, _ = temporal_train_test_split(df)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values

    # TPE sampler is the default — smarter than random or grid search
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_params = study.best_params
    best_rmse = study.best_value

    print(f"Best CV RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}")

    # Log the study results to MLflow
    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_rmse", best_rmse)
        mlflow.log_metric("n_trials", n_trials)

    return best_params


if __name__ == "__main__":
    mlflow.set_experiment("agri-yield-training")
    best = run_tuning(n_trials=50)

    # Feed best params directly into the training script
    from training.train import train

    train(params=best)

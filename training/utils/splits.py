import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "week_start",
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time: oldest rows for training, newest for test.
    Never shuffles. Never leaks future into past.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """
    Returns a TimeSeriesSplit for use in cross-validation.
    Always keeps time ordering — no shuffle.
    """
    return TimeSeriesSplit(n_splits=n_splits)

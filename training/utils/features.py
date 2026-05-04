import pandas as pd

NON_FEATURE_COLS = ["yield_kg_per_ha", "week_start", "field_id"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLS]

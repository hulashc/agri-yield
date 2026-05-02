import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Entity dataframe: prediction event on an earlier date
entity_df = pd.DataFrame(
    {
        "field_id": ["field_001"],
        "event_timestamp": [pd.Timestamp("2026-05-10 00:00:00")],
    }
)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "soil_features:soil_temp_mean",
        "vegetation_features:latest_ndvi",
        "weather_features:precip_total",
    ],
).to_df()

print(training_df)

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Bool
from feast.value_type import ValueType

field = Entity(name="field_id", join_keys=["field_id"], value_type=ValueType.STRING)

# Entity: the object you make predictions for
field = Entity(name="field_id", join_keys=["field_id"])

# Data sources
soil_source = FileSource(
    path="data/features/soil_weekly_features",
    timestamp_field="week_start",
)

vegetation_source = FileSource(
    path="data/features/vegetation_weekly_features",
    timestamp_field="week_start",
)

weather_source = FileSource(
    path="data/features/weather_weekly_features",
    timestamp_field="week_start",
)

# Feature view: soil
soil_features = FeatureView(
    name="soil_features",
    entities=[field],
    ttl=timedelta(weeks=8),
    schema=[
        Field(name="soil_temp_mean", dtype=Float32),
        Field(name="soil_temp_min", dtype=Float32),
        Field(name="soil_temp_max", dtype=Float32),
        Field(name="soil_temp_std", dtype=Float32),
        Field(name="moisture_mean", dtype=Float32),
        Field(name="moisture_std", dtype=Float32),
        Field(name="ph_mean", dtype=Float32),
        Field(name="nitrogen_mean", dtype=Float32),
        Field(name="phosphorus_mean", dtype=Float32),
        Field(name="potassium_mean", dtype=Float32),
    ],
    source=soil_source,
)

# Feature view: vegetation
vegetation_features = FeatureView(
    name="vegetation_features",
    entities=[field],
    ttl=timedelta(weeks=8),
    schema=[
        Field(name="latest_ndvi", dtype=Float32),
        Field(name="cloud_cover_pct", dtype=Float32),
        Field(name="ndvi_interpolated", dtype=Bool),
        Field(name="ndvi_proxied", dtype=Bool),
    ],
    source=vegetation_source,
)

# Feature view: weather
weather_features = FeatureView(
    name="weather_features",
    entities=[field],
    ttl=timedelta(weeks=8),
    schema=[
        Field(name="air_temp_mean", dtype=Float32),
        Field(name="precip_total", dtype=Float32),
        Field(name="humidity_mean", dtype=Float32),
        Field(name="wind_speed_mean", dtype=Float32),
    ],
    source=weather_source,
)

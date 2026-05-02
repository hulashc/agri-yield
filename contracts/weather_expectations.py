# contracts/weather_expectations.py
"""
Great Expectations validation suite for weather event data.
Run this after each weather poller writes a batch to storage.
"""

import great_expectations as gx
import pandas as pd


def build_weather_suite(context):
    """Define and save the expectation suite for weather feed readings."""

    suite = context.suites.add_or_update(
        gx.ExpectationSuite(name="weather_events_suite")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="station_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="field_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="timestamp")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="temperature", min_value=-30, max_value=55, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="precipitation", min_value=0, max_value=300, mostly=0.99
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="humidity", min_value=0, max_value=100, mostly=0.97
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="wind_speed", min_value=0, max_value=150, mostly=0.97
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeIncreasing(
            column="timestamp", strictly=True
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="temperature")
    )

    return suite


def validate_batch(df: pd.DataFrame) -> bool:
    """
    Run the weather expectation suite against a DataFrame batch.
    Returns True if valid, False if any critical check fails.
    Quarantines failed rows to a separate file.
    """
    context = gx.get_context()

    datasource = context.data_sources.add_or_update_pandas("weather_data")
    asset = datasource.add_dataframe_asset("weather_batch")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch_def")

    try:
        suite = context.suites.get("weather_events_suite")
    except Exception:
        suite = build_weather_suite(context)

    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(
            name="weather_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    result = validation_definition.run(batch_parameters={"dataframe": df})

    if not result.success:
        import os

        os.makedirs("data/quarantine", exist_ok=True)
        quarantine_path = f"data/quarantine/weather_{pd.Timestamp.now().isoformat().replace(':', '-')}.parquet"
        df.to_parquet(quarantine_path)
        print(f"[QUARANTINE] Batch failed validation. Written to {quarantine_path}")
        return False

    print(f"[VALID] Batch passed validation. {len(df)} rows accepted.")
    return True

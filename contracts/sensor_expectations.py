# contracts/sensor_expectations.py
"""
Great Expectations validation suite for soil sensor data.
Run this after each Kafka consumer writes a batch to storage.
"""

import great_expectations as gx
import pandas as pd


def build_sensor_suite(context):
    """Define and save the expectation suite for soil sensor readings."""

    # FIXED: context.suites.add_or_update() — old method no longer exists
    suite = context.suites.add_or_update(gx.ExpectationSuite(name="soil_sensors_suite"))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="field_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="device_id")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="timestamp")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="temperature", min_value=0, max_value=50, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="moisture", min_value=0, max_value=100, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ph", min_value=4.0, max_value=9.0, mostly=0.98
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="nitrogen", min_value=0, max_value=500, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="phosphorus", min_value=0, max_value=200, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="potassium", min_value=0, max_value=600, mostly=0.95
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeIncreasing(
            column="timestamp", strictly=True
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="fault_mode", value_set=["NONE", "DROPOUT", "DRIFT", "BURST"]
        )
    )

    return suite


def validate_batch(df: pd.DataFrame) -> bool:
    """
    Run the sensor expectation suite against a DataFrame batch.
    Returns True if valid, False if any critical check fails.
    Quarantines failed rows to a separate file.
    """
    context = gx.get_context()

    datasource = context.data_sources.add_or_update_pandas("sensor_data")
    asset = datasource.add_dataframe_asset("sensor_batch")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch_def")

    # FIXED: pass context so both functions share the same EphemeralDataContext
    try:
        suite = context.suites.get("soil_sensors_suite")
    except Exception:
        suite = build_sensor_suite(context)

    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(
            name="sensor_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    result = validation_definition.run(batch_parameters={"dataframe": df})

    if not result.success:
        import os

        os.makedirs("data/quarantine", exist_ok=True)
        quarantine_path = f"data/quarantine/sensors_{pd.Timestamp.now().isoformat().replace(':', '-')}.parquet"
        df.to_parquet(quarantine_path)
        print(f"[QUARANTINE] Batch failed validation. Written to {quarantine_path}")
        return False

    print(f"[VALID] Batch passed validation. {len(df)} rows accepted.")
    return True

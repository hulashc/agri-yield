# contracts/ndvi_expectations.py
"""Expectation suite for satellite NDVI data."""


def validate_ndvi_batch(df):
    """Key checks for NDVI data quality."""
    import great_expectations as gx

    context = gx.get_context()
    datasource = context.sources.add_or_update_pandas("ndvi_data")
    asset = datasource.add_dataframe_asset("ndvi_batch")
    batch_request = asset.build_batch_request(dataframe=df)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="ndvi_suite",
    )

    # NDVI must be in the physically valid range [-1, 1]
    validator.expect_column_values_to_be_between(
        "ndvi", min_value=-1.0, max_value=1.0, mostly=0.99
    )

    # Cloud cover must be a percentage
    validator.expect_column_values_to_be_between(
        "cloud_cover_pct", min_value=0, max_value=100
    )

    # Flag fields must be booleans
    validator.expect_column_values_to_be_in_set(
        "ndvi_interpolated", value_set=[True, False]
    )
    validator.expect_column_values_to_be_in_set("ndvi_proxied", value_set=[True, False])

    validator.save_expectation_suite(discard_failed_expectations=False)

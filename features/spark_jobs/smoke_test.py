from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("agri-smoke-test").getOrCreate()

# Replace with your landing path from Phase 1
input_path = "data/landing/soil-sensors/*.parquet"
output_path = "data/features/smoke_test_output"

# Read raw landing-zone data
try:
    df = spark.read.parquet(input_path)
    row_count = df.count()
    print(f"Rows read: {row_count}")

    # Tiny transform just to prove write path works
    df.limit(100).write.mode("overwrite").parquet(output_path)
    print(f"Wrote smoke test output to {output_path}")
except Exception as e:
    print(f"Smoke test failed: {e}")
    raise
finally:
    spark.stop()

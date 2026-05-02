from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("ndvi-weekly-features").getOrCreate()

input_path = "data/landing/satellite-ndvi/*.parquet"
output_path = "data/features/vegetation_weekly_features"

ndvi_df = spark.read.parquet(input_path)

ndvi_df = ndvi_df.withColumn(
    "event_time", F.to_timestamp((F.col("timestamp") / 1000).cast("double"))
).withColumn("week_start", F.date_trunc("week", F.col("event_time")))

# Get latest scene per field-week
window_latest = Window.partitionBy("field_id", "week_start").orderBy(
    F.col("event_time").desc()
)
ndvi_latest = (
    ndvi_df.withColumn("rn", F.row_number().over(window_latest))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Simple interpolation placeholder using previous value
window_prev = Window.partitionBy("field_id").orderBy("week_start")
ndvi_features = (
    ndvi_latest.withColumn("prev_ndvi", F.lag("ndvi").over(window_prev))
    .withColumn(
        "ndvi_filled",
        F.when(F.col("ndvi").isNull(), F.col("prev_ndvi")).otherwise(F.col("ndvi")),
    )
    .withColumn(
        "ndvi_interpolated",
        F.when(F.col("ndvi").isNull(), F.lit(True)).otherwise(
            F.col("ndvi_interpolated")
        ),
    )
    .withColumn("ndvi_proxied", F.coalesce(F.col("ndvi_proxied"), F.lit(False)))
    .select(
        "field_id",
        "week_start",
        F.col("ndvi_filled").alias("latest_ndvi"),
        "cloud_cover_pct",
        "ndvi_interpolated",
        "ndvi_proxied",
    )
)

ndvi_features.write.mode("overwrite").parquet(output_path)
spark.stop()

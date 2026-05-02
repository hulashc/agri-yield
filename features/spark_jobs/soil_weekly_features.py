from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("soil-weekly-features").getOrCreate()

input_path = "data/landing/soil-sensors/*.parquet"
output_path = "data/features/soil_weekly_features"

# Read raw sensor data
soil_df = spark.read.parquet(input_path)

# Convert timestamp from milliseconds to timestamp type
soil_df = soil_df.withColumn(
    "event_time", F.to_timestamp((F.col("timestamp") / 1000).cast("double"))
)

# Derive week start (Monday)
soil_df = soil_df.withColumn("week_start", F.date_trunc("week", F.col("event_time")))

weekly_soil = soil_df.groupBy("field_id", "week_start").agg(
    F.mean("temperature").alias("soil_temp_mean"),
    F.min("temperature").alias("soil_temp_min"),
    F.max("temperature").alias("soil_temp_max"),
    F.stddev("temperature").alias("soil_temp_std"),
    F.mean("moisture").alias("moisture_mean"),
    F.min("moisture").alias("moisture_min"),
    F.max("moisture").alias("moisture_max"),
    F.stddev("moisture").alias("moisture_std"),
    F.mean("ph").alias("ph_mean"),
    F.mean("nitrogen").alias("nitrogen_mean"),
    F.mean("phosphorus").alias("phosphorus_mean"),
    F.mean("potassium").alias("potassium_mean"),
)

weekly_soil.write.mode("overwrite").parquet(output_path)
spark.stop()

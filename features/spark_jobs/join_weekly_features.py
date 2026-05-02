# features/spark_jobs/join_weekly_features.py
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("join-weekly-features").getOrCreate()

soil = spark.read.parquet("data/features/soil_weekly_features")
weather = spark.read.parquet("data/features/weather_weekly_features")
veg = spark.read.parquet("data/features/vegetation_weekly_features")
labels = spark.read.parquet("data/labels/yield_labels")  # ← your label source

weekly_features = (
    soil.join(weather, on=["field_id", "week_start"], how="left")
    .join(veg, on=["field_id", "week_start"], how="left")
    .join(labels, on=["field_id", "week_start"], how="left")  # ← adds yield_kg_per_ha
)

weekly_features.write.mode("overwrite").parquet("data/features/weekly_field_features")
spark.stop()

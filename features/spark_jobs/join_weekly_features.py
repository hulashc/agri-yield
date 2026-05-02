from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("join-weekly-features").getOrCreate()

soil = spark.read.parquet("data/features/soil_weekly_features")
weather = spark.read.parquet("data/features/weather_weekly_features")
veg = spark.read.parquet("data/features/vegetation_weekly_features")

weekly_features = soil.join(weather, on=["field_id", "week_start"], how="left").join(
    veg, on=["field_id", "week_start"], how="left"
)

weekly_features.write.mode("overwrite").parquet("data/features/weekly_field_features")
spark.stop()

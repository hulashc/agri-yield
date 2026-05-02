from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("weather-weekly-features").getOrCreate()

input_path = "data/landing/weather-events/*.parquet"
output_path = "data/features/weather_weekly_features"

weather_df = spark.read.parquet(input_path)

weather_df = weather_df.withColumn(
    "event_time", F.to_timestamp((F.col("timestamp") / 1000).cast("double"))
).withColumn("week_start", F.date_trunc("week", F.col("event_time")))

# Fill missing hourly weather values with simple forward-fill approximation
weather_df = weather_df.fillna(
    {
        "precipitation": 0.0,
    }
)

weekly_weather = weather_df.groupBy("field_id", "week_start").agg(
    F.mean("temperature").alias("air_temp_mean"),
    F.sum("precipitation").alias("precip_total"),
    F.mean("humidity").alias("humidity_mean"),
    F.mean("wind_speed").alias("wind_speed_mean"),
)

weekly_weather.write.mode("overwrite").parquet(output_path)
spark.stop()

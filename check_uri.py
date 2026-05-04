import mlflow
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
client = mlflow.MlflowClient()
versions = client.search_model_versions("name='agri-yield-xgb'")
for v in versions:
    print(f"v{v.version} stage={v.current_stage} source={v.source}")

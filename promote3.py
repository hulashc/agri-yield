import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
versions = client.search_model_versions("name='agri-yield-xgb'")
latest = sorted(versions, key=lambda v: int(v.version))[-1]
print("Promoting version:", latest.version)
client.transition_model_version_stage("agri-yield-xgb", latest.version, "Production")
print("Done")

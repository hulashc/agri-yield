import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
versions = client.search_model_versions("name='agri-yield-xgb'")
for v in versions:
    print(v.version, v.current_stage, v.status)
if not versions:
    print("NO VERSIONS FOUND")

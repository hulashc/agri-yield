import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
v = client.get_model_version("agri-yield-xgb", "1")
artifacts = client.list_artifacts(v.run_id)
for a in artifacts:
    print(a.path)

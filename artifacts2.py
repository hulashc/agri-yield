import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
v = client.get_model_version("agri-yield-xgb", "1")
print("Run ID:", v.run_id)
print("Source:", v.source)


def list_all(run_id, path=""):
    arts = client.list_artifacts(run_id, path)
    for a in arts:
        print(a.path, "(dir)" if a.is_dir else "")
        if a.is_dir:
            list_all(run_id, a.path)


list_all(v.run_id)

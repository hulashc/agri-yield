import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
client.transition_model_version_stage("agri-yield-xgb", "1", "Production")
print("Promoted to Production")

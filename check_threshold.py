import sys
import os
import mlflow

# 1. Explicitly connect to DagsHub
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# 2. Read the Run ID from the artifact
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking accuracy for Run ID: {run_id}")
except FileNotFoundError:
    print("Error: model_info.txt not found. Did the download-artifact step work?")
    sys.exit(1)

# 3. Fetch the metrics from MLflow
try:
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0.0)
    print(f"Accuracy retrieved from MLflow: {accuracy}")
except Exception as e:
    print(f"Error fetching data from MLflow: {e}")
    sys.exit(1)

# 4. Enforce the Threshold
if accuracy < 0.85:
    print("❌ Deployment halted! Model accuracy is below the 0.85 threshold.")
    sys.exit(1)
else:
    print("✅ Model passed validation! Proceeding to deployment...")
    sys.exit(0)
import sys
import mlflow

# 1. Read the Run ID from the artifact
try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    print(f"Checking accuracy for Run ID: {run_id}")
except FileNotFoundError:
    print("Error: model_info.txt not found. Did the download-artifact step work?")
    sys.exit(1)

# 2. Fetch the metrics from MLflow
run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)
print(f"Accuracy retrieved from MLflow: {accuracy}")

# 3. Enforce the Threshold
if accuracy < 0.85:
    print("Deployment halted! Model accuracy is below the 0.85 threshold.")
    sys.exit(1) # This command explicitly tells GitHub Actions to fail the job
else:
    print("Model passed validation! Proceeding to deployment...")
    sys.exit(0)
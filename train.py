import mlflow
import argparse
import os

# Argument parser to force a fail for your screenshot
parser = argparse.ArgumentParser()
parser.add_argument("--fail", action="store_true", help="Force accuracy below 0.85")
args = parser.parse_args()

accuracy = 0.80 if args.fail else 0.92

# Ensure MLflow knows to point to the remote DagsHub server
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model trained. Logged accuracy: {accuracy}")
    
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"Run ID {run.info.run_id} successfully written to model_info.txt")
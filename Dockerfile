# 1. Use the specified base image
FROM python:3.10-slim

# 2. Accept the MLflow Run ID as an argument
ARG RUN_ID
ENV MODEL_RUN_ID=$RUN_ID

# 3. Simulate downloading the model weights
RUN echo "Successfully connected to MLflow. Downloading model weights for Run ID: ${MODEL_RUN_ID}..."

# 4. Mock complete
RUN echo "Mock Docker container successfully built!"

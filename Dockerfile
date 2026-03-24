# 1. Starts from a lightweight Python base image
FROM python:3.10-slim

ARG RUN_ID
ENV MODEL_RUN_ID=$RUN_ID

# Simulate downloading the model weights using the Run ID
RUN echo "Successfully connected to MLflow. Downloading model weights for Run ID: ${MODEL_RUN_ID}..."

# Sets the WORKDIR inside the container
WORKDIR /app

# We copy ONLY the requirements, Docker will cache this layer so it doesn't have to re-download PyTorch every single time change  python script
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the code 
COPY simple_gan.py .
COPY simple_digits.csv .

# 4. Defines a CMD to run the training script automatically
CMD ["python", "simple_gan.py"]
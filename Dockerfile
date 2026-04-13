# ── Base image – slim Python 3.10 keeps the layer small ──────────────────────
FROM python:3.10-slim

# Metadata
LABEL maintainer="mlops-assignment-6"
LABEL description="GAN training container with PyTorch CPU and MLflow tracking"

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies required by certain Python packages ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .

# Install everything, honouring the PyTorch CPU extra-index-url
# that is already declared inside requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY train.py          .
COPY check_threshold.py .
COPY simple_gan.py     .

# Accept an MLflow Run ID at build time (used by the deploy step)
ARG RUN_ID=""
ENV RUN_ID=${RUN_ID}

# MLflow tracking credentials are injected at runtime via -e / --env-file
# ENV MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "train.py"]
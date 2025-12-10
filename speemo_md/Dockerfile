# 1) Choose a slim CUDA-enabled base
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2) Install minimal system deps (single layer)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential git curl libsndfile1 wget && \
    rm -rf /var/lib/apt/lists/*

# 3) Install Python 3.10 and pip, then upgrade pip, setuptools, and wheel
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# 4) Copy requirements and install Python packages with retry
WORKDIR /workspace
COPY requirements.txt .
RUN pip cache purge && \
    bash -c "\
      for i in 1 2 3; do \
        pip install --no-cache-dir -r requirements.txt && break || \
        (echo 'Retrying pip install...' && sleep 5); \
      done"

# 5) Copy project source code (excluding large dirs via .dockerignore)
COPY . .

# 6) Create and switch to non-root user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /workspace
USER appuser

# 7) Declare external volumes for large datasets/models
VOLUME ["/workspace/data", "/workspace/models", "/workspace/Archive", "/workspace/logs", "/workspace/docs", "/workspace/links"]

# 8) Default entrypoint for your application
ENTRYPOINT ["python", "run.py"]

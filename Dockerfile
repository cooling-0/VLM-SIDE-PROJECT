# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# --- basic envs ---
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     MPLBACKEND=Agg     DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# --- system deps (minimal, headless) ---
RUN apt-get update && apt-get install -y --no-install-recommends     git ca-certificates curl     && rm -rf /var/lib/apt/lists/*

# --- python deps ---
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip &&     pip install -r /tmp/requirements.txt

# for notebooks & outputs
RUN mkdir -p /workspace/notebooks /workspace/data/images /workspace/out

# Default command: Jupyter Lab (no token; local dev)
EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''"]

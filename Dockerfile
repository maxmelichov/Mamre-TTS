# Use NVIDIA PyTorch base image with CUDA support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Avoid hardlink warning when cache and .venv are on different filesystems
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    wget \
    curl \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    festival \
    software-properties-common \
    gnupg2 \
    lsb-release \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock ./

# Bootstrap venv and install deps (same logic as install.sh; no uv lock in image)
RUN uv venv \
    && uv pip install setuptools wheel hatchling psutil numpy \
    && uv pip install --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.6.0+cu124' 'torchaudio==2.6.0+cu124' \
    && uv sync

# Optional: DiffMamba backend (clone into project or mount)
COPY DiffMamba /tmp/DiffMamba
RUN uv pip install --no-build-isolation -e /tmp/DiffMamba

# Copy the rest of the project
COPY . .

RUN mkdir -p /app/outputs && chmod +x inference.py

EXPOSE 7860 8000

CMD ["uv", "run", "python", "inference.py"]

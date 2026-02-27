# Use NVIDIA PyTorch base image with CUDA support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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
    && rm -rf /var/lib/apt/lists/*


# Update package list and install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# Install pip for Python 3.10 (if it's not already installed)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel ninja

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0

# Install flash-attn and causal-conv1d with specific installation method (without altering torch version)
# Pre-install build/runtime deps needed by flash-attn metadata
RUN pip install --no-cache-dir numpy psutil packaging cmake pybind11
RUN pip install --no-build-isolation --prefer-binary --no-cache-dir --no-deps \
    flash-attn==2.7.4.post1
RUN pip install --no-build-isolation --prefer-binary --no-cache-dir --no-deps \
    causal-conv1d==1.5.0.post8 

# Copy requirements file first
COPY requirements.txt .

# Install all requirements together to ensure compatibility
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# Install git-based packages
RUN pip install phonikud-onnx
RUN pip install git+https://github.com/thewh1teagle/phonikud

COPY DiffMamba /tmp/DiffMamba
RUN pip install --no-build-isolation /tmp/DiffMamba
# RUN rm -rf /tmp/DiffMamba

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/outputs

# Set permissions
RUN chmod +x inference.py

# Expose ports (7860 for Gradio, 8000 for FastAPI)
EXPOSE 7860 8000

# Set the default command
CMD ["python3", "inference.py"]
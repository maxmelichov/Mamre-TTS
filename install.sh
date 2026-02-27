#!/usr/bin/env bash
# Bootstrap venv with setuptools, wheel, hatchling, psutil, numpy, and torch 2.6+cu124
# so flash-attn/causal-conv1d and phonikud (git) can be built with --no-build-isolation.
# Torch must be 2.6.0+cu124 so prebuilt wheels for flash-attn/causal-conv1d are used.
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
# Avoid hardlink warning when cache and .venv are on different filesystems
export UV_LINK_MODE=copy
uv venv
uv pip install setuptools wheel hatchling psutil numpy
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.6.0+cu124' 'torchaudio==2.6.0+cu124'
uv lock
uv sync
# DiffMamba backend (required for inference/training)
if [ ! -d "DiffMamba" ]; then
  git clone https://github.com/maxmelichov/DiffMamba.git
fi
cd DiffMamba && uv pip install -e . && cd ..

# Verify torch 2.6 with CUDA
uv run python -c "
import torch
v = torch.__version__
assert v.startswith('2.6.'), f'Expected torch 2.6.x, got {v}'
assert 'cu124' in v or 'cu' in v, f'Expected CUDA build, got {v}'
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
"

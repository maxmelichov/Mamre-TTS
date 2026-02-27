#!/usr/bin/env bash
# Bootstrap venv with setuptools, wheel, hatchling, psutil, numpy, and torch so flash-attn/causal-conv1d
# and phonikud (git) can be built with --no-build-isolation.
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
uv venv
uv pip install setuptools wheel hatchling psutil numpy torch torchaudio
uv sync

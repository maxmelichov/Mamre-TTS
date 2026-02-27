# MamreStream Hebrew TTS

Hebrew text-to-speech with real-time streaming and GPU acceleration. Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

## Prerequisites

- **GPU:** 6GB+ VRAM (PyTorch 2.6 + CUDA 12.4)
- **System (e.g. Ubuntu):** `espeak-ng`, `ffmpeg`, `libsndfile1`, `build-essential`, `python3-dev`

## Setup

1. Install [uv](https://docs.astral.sh/uv/), then clone the repo:

```bash
git clone <this-repo>
cd Mamre-TTS
```

2. Run the installer. It creates a venv, installs **PyTorch 2.6.0+cu124**, project deps (including flash-attn, causal-conv1d), clones [DiffMamba](https://github.com/maxmelichov/DiffMamba) if missing, and installs it in editable mode. At the end it checks that torch 2.6 with CUDA is available.

```bash
./install.sh
```

No need to clone DiffMamba manually—the script does it if `DiffMamba/` is not present.

## Model & weights

Place under `./weights`:

- `MamreV1_3_epoch3.pt` — TTS model
- `dac_44khz.safetensors` or `dac_44khz.pt` — autoencoder
- `phonikud-1.0.onnx`, `dictabert_tokenizer.json` — Phonikud diacritization

No Hugging Face downloads at runtime; missing files → model load fails.

**Model weights are for non-commercial use only.** See [License](#license).

## Scripts

```bash
uv run python inference.py --text "טקסט" --target_speaker_path voices/AUD-20251102-WA0012.mp3
uv run python stream.py
```

## Training

Single GPU (uses `cuda:0`; autoencoder can use `cuda:1` if present):

```bash
uv run python train.py --csv_file data/your_data.csv --save_dir checkpoints --epochs 5 --batch_size 2
```

Multi-GPU: the script uses all visible CUDA devices automatically. With 2+ GPUs the model is wrapped in `DataParallel` and checkpoints are saved **without** the `module.` prefix, so they load directly for single-GPU inference.

To use only specific GPUs (e.g. 2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python train.py --csv_file data/your_data.csv --save_dir checkpoints
```

Resume from a checkpoint:

```bash
uv run python train.py --csv_file data/your_data.csv --checkpoint checkpoints/train_epoch_3.pt --save_dir checkpoints
```

CSV must have columns `filename` (path to WAV) and `phonemes` (transcription); optional `quality` column (float) for per-sample weighting.

## License

**Code and model architecture:** Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

**Model weights** (e.g. files in `./weights`): **non-commercial use only.** You may not use the provided weights for commercial purposes.

Upstream components may have their own licenses.

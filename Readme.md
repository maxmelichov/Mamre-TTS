# Mamre Hebrew TTS

Hebrew text-to-speech with streaming support and GPU acceleration. Uses [uv](https://docs.astral.sh/uv/) for dependencies.

## Prerequisites

- **GPU:** 6GB+ VRAM (PyTorch 2.6 + CUDA 12.4)
- **System (Ubuntu):** `espeak-ng`, `ffmpeg`, `libsndfile1`, `build-essential`, `python3-dev`

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and clone the repo:

   ```bash
   git clone <this-repo>
   cd Mamre-TTS
   ```

2. Run the installer (creates venv, installs PyTorch + deps, pulls DiffMamba if needed):

   ```bash
   ./install.sh
   ```

## Model

Weights are loaded from **[notmax123/MamreTTS](https://huggingface.co/notmax123/MamreTTS)** on Hugging Face. The first run of inference or training will download the model.

Model weights are **non-commercial use only**. See [License](#license).

## Usage

**Synthesize from text** (optional: add `-o out.wav` or `-i script.txt`):

```bash
uv run python inference.py --text "טקסט לעיבוד" -v voices/your_voice.wav
```

**Streaming TTS** (reads built-in Hebrew text, writes `streaming.wav`):

```bash
uv run python stream.py -v voices/your_voice.wav
```

**Train on your data** (CSV with columns `filename`, `phonemes`; optional `quality`):

```bash
uv run python train.py --csv_file data/tts_data.csv --save_dir checkpoints --epochs 5
```

Resume from a checkpoint:

```bash
uv run python train.py --csv_file data/tts_data.csv --checkpoint checkpoints/train_epoch_3.pt --save_dir checkpoints
```

With multiple GPUs, the script uses all visible CUDA devices. Restrict with `CUDA_VISIBLE_DEVICES=0,1`.

## License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE).
- **Model weights:** Non-commercial use only.

Upstream components may have separate licenses.

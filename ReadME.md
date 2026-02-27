# MamreStream Hebrew TTS

Hebrew text-to-speech with real-time streaming and GPU acceleration.

## Prerequisites

- GPU: 6GB+ VRAM
- [DiffMamba](https://github.com/maxmelichov/DiffMamba) backend (clone into project root)

## Setup

Install [uv](https://docs.astral.sh/uv/), then clone the repo and DiffMamba:

```bash
git clone <this-repo>
cd Mamre-TTS
git clone https://github.com/maxmelichov/DiffMamba.git
```

Run the installer (creates venv, installs PyTorch CUDA, and project deps):

```bash
./install.sh
```

Install the DiffMamba backend:

```bash
cd DiffMamba && uv pip install -e . && cd ..
```

System deps (e.g. Ubuntu): `espeak-ng`, `ffmpeg`, `libsndfile1`, `build-essential`, `python3-dev`.

## Model & weights

Place under `./weights`:

- `MamreV1_3_epoch3.pt` — TTS model
- `dac_44khz.safetensors` or `dac_44khz.pt` — autoencoder
- `phonikud-1.0.onnx`, `dictabert_tokenizer.json` — Phonikud diacritization

No Hugging Face downloads at runtime; missing files → model load fails.

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

See upstream licenses for components.

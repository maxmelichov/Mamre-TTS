# MamreStream Hebrew TTS

Hebrew text-to-speech with real-time streaming and GPU acceleration.

## Prerequisites

- GPU: 6GB+ VRAM
- [DiffMamba](https://github.com/maxmelichov/DiffMamba) backend (clone into project root)

## Setup

Install [uv](https://docs.astral.sh/uv/) then:

```bash
git clone <this-repo>
cd Mamre-TTS
git clone https://github.com/maxmelichov/DiffMamba.git

./install.sh
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

## License

See upstream licenses for components.

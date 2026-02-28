#!/usr/bin/env python
"""
Long Text Inference Script for Mamre-v1 with segment concatenation and
re-encoding for tail-frame continuity.
"""

import argparse
import torch
import torchaudio
import re
import os
from pydub import AudioSegment
from tqdm import tqdm
from typing import List
from Mamre.model import Mamre
from Mamre.conditioning import make_cond_dict, is_hebrew_text


def split_text_into_segments(text: str) -> List[str]:
    """Split text into segments by newline characters."""
    segments = re.split(r'[\n]', text)
    return [seg.strip() for seg in segments if seg.strip()]


DEFAULT_TEXT = """
        jeʁuʃalˈajim hˈi ʔˈiʁ ʔatikˈa vaχaʃuvˈa bimjuχˈad, ʃemeχilˈa betoχˈa ʃχavˈot ʁabˈot ʃˈel histˈoʁja,
        taʁbˈut veʁuχanijˈut ʃenimʃaχˈot ʔalfˈej ʃanˈim, vehˈi mehavˈa mokˈed meʁkazˈi liʃlˈoʃet
        hadatˈot haɡdolˈot, jahadˈut, natsʁˈut veʔislˈam.
        ʃemitχabʁˈot jˈaχad bemakˈom ʔeχˈad jiχudˈi, malˈe ʔenˈeʁɡija umuʁkavˈut,
        ʃˈam ʔefʃˈaʁ limtsˈo ʔataʁˈim kdoʃˈim, ʃχunˈot ʔatikˈot, veʃvakˈim tsivʔonijˈim,
        vekˈol pinˈa mesapˈeʁet sipˈuʁ ʃˈel tkufˈot ʃonˈot, ʔanaʃˈim ʃonˈim veʔejʁuʔˈim ʃehiʃpˈiʔu ʔˈal hahistˈoʁja ʃˈel haʔolˈam kulˈo,
        mˈa ʃehofˈeχ ʔet jeʁuʃalˈajim lˈo ʁˈak leʔˈiʁ ɡeʔoɡʁˈafit ʔˈela ɡˈam lemeʁkˈaz ʃˈel zehˈut,
        ʔemunˈa vezikaʁˈon kolektˈivi ʃemamʃˈiχ leʔoʁˈeʁ haʃʁaʔˈa ulχabˈeʁ bˈen ʔanaʃˈim meʁˈeka ʃonˈe mikˈol kitsvˈot tevˈel.
         """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mamre-v1 long-text TTS with segment concatenation and tail-frame continuity."
    )
    parser.add_argument("--voice", "-v", type=str, default="voices/Female1.mp3",
                        help="Path to reference audio for speaker embedding.")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/train_epoch_1.pt",
                        help="Optional checkpoint to load (fine-tuned weights).")
    parser.add_argument("--output", "-o", type=str, default="male1_hebrew.wav",
                        help="Output WAV path for combined audio.")
    parser.add_argument("--segments_dir", type=str, default="generated_segments",
                        help="Directory to save per-segment WAVs.")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize (overrides --input_file if set).")
    parser.add_argument("--input_file", "-i", type=str, default=None,
                        help="Path to text file to synthesize (used if --text is not set).")
    parser.add_argument("--model_id", type=str, default="notmax123/MamreTTS",
                        help="HuggingFace model repo id.")
    parser.add_argument("--model_filename", type=str, default="MamreV1.safetensors",
                        help="Model filename in the repo.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. cuda:0 or cpu). Default: cuda:0 if available else cpu.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--language", type=str, default="he",
                        help="Language code for conditioning (e.g. he, en-us).")
    parser.add_argument("--speaking_rate", type=float, default=9.0,
                        help="Speaking rate (phonemes per minute; ~10 slow, ~30 fast).")
    parser.add_argument("--tail_frames", type=int, default=8,
                        help="Number of code frames to carry over between segments for continuity.")
    parser.add_argument("--min_tokens", type=int, default=200,
                        help="Minimum max_new_tokens (floor for segment length).")
    parser.add_argument("--min_length_ratio", type=float, default=3.5,
                        help="Segment min length = max(min_len, len(segment) * this).")
    parser.add_argument("--max_tokens_ratio", type=float, default=6.5,
                        help="Segment max_new_tokens = max(min_tokens, len(segment) * this).")
    parser.add_argument("--min_len_short", type=int, default=20,
                        help="Min length for short segments (when len*2 < 150).")
    parser.add_argument("--min_len_long", type=int, default=100,
                        help="Min length for long segments (when len*2 >= 150).")
    parser.add_argument("--min_p", type=float, default=0.1,
                        help="Min-p sampling threshold.")
    parser.add_argument("--repetition_penalty", type=float, default=3.0,
                        help="Repetition penalty for generation.")
    parser.add_argument("--repetition_penalty_window", type=int, default=64,
                        help="Sliding window size for repetition penalty.")
    parser.add_argument("--silence_comma_ms", type=int, default=50,
                        help="Silence (ms) after comma between segments.")
    parser.add_argument("--silence_period_ms", type=int, default=50,
                        help="Silence (ms) after period between segments.")
    parser.add_argument("--crossfade_ms", type=int, default=20,
                        help="Crossfade duration (ms) between segment boundaries.")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    os.makedirs(args.segments_dir, exist_ok=True)

    print(f"Loading model from {args.model_id}...")
    try:
        model = Mamre.from_pretrained(args.model_id, model_filename=args.model_filename, device="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}. Ensure the model can be downloaded or is cached.") from e

    if os.path.isfile(args.checkpoint):
        try:
            state = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("Checkpoint loaded.")
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")
    elif args.checkpoint:
        print(f"Checkpoint not found ({args.checkpoint}), using base model only.")

    model.eval()
    model.to(device)
    model.autoencoder.dac.eval()
    model.autoencoder.dac.to(device)

    if not os.path.isfile(args.voice):
        raise FileNotFoundError(f"Speaker audio not found: {args.voice}")
    wav, sr = torchaudio.load(args.voice)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, model.autoencoder.sampling_rate)
    sr = model.autoencoder.sampling_rate
    print("Computing speaker embedding...")
    with torch.no_grad():
        speaker = model.make_speaker_embedding(wav, sr)

    if args.text is not None:
        long_text = args.text
    elif args.input_file:
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError(f"Input text file not found: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            long_text = f.read()
    else:
        long_text = DEFAULT_TEXT
    segments = split_text_into_segments(long_text)
    print(f"Split text into {len(segments)} segments")

    all_generated = []
    prev_tail_codes = None
    sampling_params = dict(
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        repetition_penalty_window=args.repetition_penalty_window,
    )

    for i, segment in enumerate(tqdm(segments, desc="Generating")):
        print(f"\nSegment {i+1}/{len(segments)}: {segment[:40]}")
        attempt = 0
        min_len = args.min_len_short if int(len(segment) * 2) < 150 else args.min_len_long
        min_length = max(min_len, int(len(segment) * args.min_length_ratio))
        max_new_tokens = max(args.min_tokens, int(len(segment) * args.max_tokens_ratio))
        while True:
            attempt += 1
            cond = make_cond_dict(
                text=segment,
                language="he" if is_hebrew_text(segment) else args.language,
                speaking_rate=args.speaking_rate,
                speaker=speaker,
            )
            conditioning = model.prepare_conditioning(cond)
            with torch.no_grad():
                if prev_tail_codes is not None:
                    codes = model.generate(
                        conditioning,
                        audio_prefix_codes=prev_tail_codes,
                        max_new_tokens=max_new_tokens,
                        sampling_params=sampling_params,
                    )
                else:
                    codes = model.generate(
                        conditioning,
                        max_new_tokens=max_new_tokens,
                        sampling_params=sampling_params,
                    )
            if codes.shape[-1] >= min_length and codes.shape[-1] <= max_new_tokens:
                break
            print(f"Segment {i+1}: generated length {codes.shape[-1]} < {min_length}, retrying (attempt {attempt})...")

        with torch.no_grad():
            wav_out = model.autoencoder.decode(codes).cpu()[0]

        if args.tail_frames:
            prev_tail_codes = codes[..., -args.tail_frames:]
        if i > 0:
            samples_per_frame = wav_out.shape[-1] // codes.shape[-1]
            tail_samples = args.tail_frames * samples_per_frame
            wav_out = wav_out[..., tail_samples:]

        seg_path = os.path.join(args.segments_dir, f"segment_{i:03d}.wav")
        torchaudio.save(seg_path, wav_out, sr)
        all_generated.append(seg_path)

    pattern = re.compile(r'([^.,]+)([.,]?)')
    segments_with_punct = [
        (m.group(1).strip(), m.group(2))
        for m in pattern.finditer(long_text)
        if m.group(1).strip()
    ]
    silence_map = []
    for text_seg, punct in segments_with_punct:
        if punct == ',':
            silence_map.append(args.silence_comma_ms)
        elif punct == '.':
            silence_map.append(args.silence_period_ms)
        else:
            silence_map.append(0)

    combined = AudioSegment.empty()
    for idx, file_path in enumerate(all_generated):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Generated segment missing: {file_path}")
        seg_audio = AudioSegment.from_file(file_path)
        seg_dur = len(seg_audio)
        if idx == 0:
            combined = seg_audio
        else:
            if seg_dur >= args.crossfade_ms:
                combined = combined.append(seg_audio, crossfade=args.crossfade_ms)
            else:
                combined = combined + seg_audio
        if idx < len(silence_map) - 1 and silence_map[idx] > 0:
            combined += AudioSegment.silent(duration=silence_map[idx])

    combined.export(args.output, format="wav")
    print(f"Saved combined audio to {args.output}")

if __name__ == "__main__":
    main()

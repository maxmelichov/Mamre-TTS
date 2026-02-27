import os
import sys
import time
import argparse

import torch
import torchaudio
import wave
import numpy as np

import re

from Mamre.model import Mamre

texts = [
    """בעידן הבינה המלאכותית, הגבולות בין אדם למכונה מיטשטשים יותר מתמיד. יכולות החישוב והלמידה של מערכות מתקדמות מאפשרות כיום למחשבים להבין שפה, ליצור תוכן, ואף לזהות רגשות או כוונות אנושיות. עם זאת, האתגר האמיתי אינו רק טכנולוגי — הוא מוסרי וחברתי. השאלה איננה "מה אפשר לבנות", אלא "מה נכון לבנות". ככל שהטכנולוגיה מתקדמת, כך גוברת האחריות של המפתחים לוודא שהיא משרתת את האנושות — ולא להפך."""
]

MAX_SEGMENT_RETRIES = int(os.getenv("MAMRE_STREAM_SEGMENT_RETRIES", "5"))
STREAM_RETRY_SLEEP = float(os.getenv("MAMRE_STREAM_RETRY_SLEEP", "0.5"))
STREAM_FALLBACK_SILENCE_MS = int(os.getenv("MAMRE_STREAM_FALLBACK_SILENCE_MS", "400"))


def split_text_into_segments(text):
    """
    Split long text into manageable segments while preserving natural boundaries.
    Mirrors the segmentation strategy used in inference.py:
    - Target upper limit: int(len(segment) * 6.75) <= 700 tokens
    - Prefer split on: newline > '.' > ',' > space
    Returns list of (segment_text, sep) where sep in {'.', ',', ' '}.
    """
    if not text:
        return []

    ratio = 6.75
    max_tokens = 700
    max_chars = int(max_tokens // ratio)  # ~103-107

    def _chunk_to_limit(s, limit):
        chunks = []
        start = 0
        n = len(s)
        while start < n:
            end = min(n, start + limit)
            window = s[start:end]
            cut_rel = -1
            chosen_sep = ' '
            for sep in ['\n', '.', ',', ' ']:
                pos = window.rfind(sep)
                if pos != -1:
                    cut_rel = pos
                    chosen_sep = '.' if sep == '.' else (',' if sep == ',' else ' ')
                    break
            if cut_rel == -1:
                cut_rel = len(window) - 1
                chosen_sep = ' '
            chunk = s[start:start + cut_rel + 1].strip()
            if chunk:
                chunks.append((chunk, chosen_sep))
            start = start + cut_rel + 1
        return chunks

    segments = []
    buf = []
    last_space = -1
    last_comma = -1
    last_period = -1

    for ch in text:
        if ch == '\n':
            if buf:
                seg = ''.join(buf).strip()
                if seg:
                    if len(seg) > max_chars:
                        segments.extend(_chunk_to_limit(seg, max_chars))
                    else:
                        segments.append((seg, ' '))
            buf = []
            last_space = last_comma = last_period = -1
            continue

        idx = len(buf)
        buf.append(ch)
        if ch == ' ':
            last_space = idx
        elif ch == ',':
            last_comma = idx
        elif ch == '.':
            last_period = idx

        if len(buf) > max_chars:
            cut_idx = -1
            for candidate in (last_period, last_comma, last_space):
                if candidate != -1 and candidate < len(buf):
                    cut_idx = candidate
                    break
            if cut_idx == -1:
                cut_idx = max_chars - 1

            seg = ''.join(buf[:cut_idx + 1]).strip()
            if seg:
                sep_char = ' '
                if cut_idx == last_period:
                    sep_char = '.'
                elif cut_idx == last_comma:
                    sep_char = ','
                elif cut_idx == last_space:
                    sep_char = ' '
                segments.append((seg, sep_char))

            buf = buf[cut_idx + 1:]
            last_space = last_comma = last_period = -1
            for i, c in enumerate(buf):
                if c == ' ':
                    last_space = i
                elif c == ',':
                    last_comma = i
                elif c == '.':
                    last_period = i

    if buf:
        seg = ''.join(buf).strip()
        if seg:
            if len(seg) > max_chars:
                segments.extend(_chunk_to_limit(seg, max_chars))
            else:
                segments.append((seg, ' '))

    out = []
    for seg_text, sep in segments:
        s = seg_text.strip()
        if s:
            out.append((s, sep if sep in {'.', ','} else ' '))
    return out


def main():
    parser = argparse.ArgumentParser(description="Stream TTS with Mamre")
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=os.getenv("MAMRE_VOICE", "voices/Female1.mp3"),
        help="Path to reference audio for speaker embedding (default: voices/reference.mp3 or MAMRE_VOICE)",
    )
    args = parser.parse_args()
    voice_path = args.voice

    if not os.path.isfile(voice_path):
        print(
            f"Error: Reference audio file not found: {voice_path}",
            file=sys.stderr,
        )
        print(
            "Provide a path to a short reference recording (e.g. WAV or MP3), e.g.:",
            file=sys.stderr,
        )
        print("  uv run stream.py --voice /path/to/your/voice.wav", file=sys.stderr)
        print("  or set MAMRE_VOICE=/path/to/voice.wav", file=sys.stderr)
        sys.exit(1)

    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model from notmax123/MamreTTS...")
    model = Mamre.from_pretrained("notmax123/MamreTTS", model_filename="MamreV1.pt", device="cpu")

    model.eval()
    model.to(device)
    model.autoencoder.dac.eval()
    model.autoencoder.dac.to(device)

    out_sr = model.autoencoder.sampling_rate
    samples_per_ms = out_sr / 1000.0

    # Load a reference speaker audio to generate a speaker embedding.
    print(f"Loading reference audio from {voice_path}...")
    wav, sr = torchaudio.load(voice_path)
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(777)

    # Streaming write setup
    wav_writer = None
    t0 = time.time()
    generated = 0
    ttfb = None
    failed_segments: list[int] = []

    # Prepare robust segments to avoid long-context degradation
    full_text = texts[0].strip()
    segments_with_seps = split_text_into_segments(full_text)
    segments = [s for s, _ in segments_with_seps]
    separators = [sep for _, sep in segments_with_seps]
    print(f"Split text into {len(segments)} segments for streaming")

    # Punctuation-aware silence insertion (ms)
    silence_map = {'.': 80, ',': 50, ' ': 0}
    def ms_to_samples(ms):
        return int(out_sr * (ms / 1000.0))

    def ensure_wav_writer(channels: int = 1):
        nonlocal wav_writer
        if wav_writer is not None:
            return
        wav_writer = wave.open("streaming.wav", "wb")
        wav_writer.setnchannels(channels)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(out_sr)

    # --- STREAMING GENERATION (per-segment reset) ---
    print("Starting per-segment streaming generation...")

    chunk_schedule = [22, 13, *range(12, 100)]
    chunk_overlap = 1

    seg_counter = 0
    for seg_idx, segment in enumerate(segments):
        if not segment.strip():
            continue

        def build_segment_generator():
            def _gen():
                elapsed = int((time.time() - t0) * 1000)
                preview = segment[:80] + ("..." if len(segment) > 80 else "")
                print(f"Yielding segment {seg_idx + 1}/{len(segments)} at {elapsed}ms: {preview}")
                yield {
                    "text": segment,
                    "speaker": speaker,
                    "language": "he",
                }
            return _gen()

        segment_success = False
        last_error = None

        for attempt in range(1, MAX_SEGMENT_RETRIES + 1):
            try:
                stream_generator = model.stream(
                    cond_dicts_generator=build_segment_generator(),
                    chunk_schedule=chunk_schedule,
                    chunk_overlap=chunk_overlap,
                    mark_boundaries=False,
                )

                for audio_chunk in stream_generator:
                    ensure_wav_writer(int(audio_chunk.shape[0]))

                    chunk = audio_chunk.detach().to(torch.float32).clamp_(-1.0, 1.0).cpu()
                    pcm16 = (chunk.numpy().T * 32767.0).astype(np.int16)
                    wav_writer.writeframes(pcm16.tobytes())

                    elapsed = int((time.time() - t0) * 1000)
                    if ttfb is None:
                        ttfb = elapsed
                    gap = "GAP" if ttfb + generated < elapsed else ""
                    generated += int(audio_chunk.shape[1] / samples_per_ms)
                    seg_counter += 1
                    print(
                        f"Chunk {seg_counter:>3}: elapsed {elapsed:>5}ms | "
                        f"generated up to {ttfb + generated:>5}ms {gap}"
                    )

                segment_success = True
                break

            except Exception as exc:
                last_error = exc
                print(
                    f"[Stream] Segment {seg_idx + 1} attempt {attempt}/{MAX_SEGMENT_RETRIES} failed: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(min(2.0, STREAM_RETRY_SLEEP * attempt))

        if not segment_success:
            failed_segments.append(seg_idx + 1)
            fallback_ms = max(STREAM_FALLBACK_SILENCE_MS, len(segment) * 35)
            fallback_samples = ms_to_samples(fallback_ms)
            ensure_wav_writer(1)
            silence = torch.zeros((1, fallback_samples), dtype=torch.float32)
            pcm16_sil = (silence.numpy().T * 32767.0).astype(np.int16)
            wav_writer.writeframes(pcm16_sil.tobytes())
            generated += int(fallback_samples / samples_per_ms)
            if ttfb is None:
                ttfb = int((time.time() - t0) * 1000)
            print(
                f"[Stream] Segment {seg_idx + 1} failed after retries ({last_error}). "
                f"Wrote {fallback_ms}ms of silence fallback."
            )

        # After finishing this segment, add punctuation-aware silence
        if seg_idx < len(separators):
            sil_ms = silence_map.get(separators[seg_idx], 0)
            sil_samples = ms_to_samples(sil_ms)
            if sil_samples > 0 and wav_writer is not None:
                silence = torch.zeros((1, sil_samples), dtype=torch.float32)
                pcm16_sil = (silence.numpy().T * 32767.0).astype(np.int16)
                wav_writer.writeframes(pcm16_sil.tobytes())
                generated += int(sil_samples / samples_per_ms)

    # Close writer if opened
    if wav_writer is not None:
        wav_writer.close()

    generation = round(time.time() - t0, 3)
    duration = round(generated / 1000.0, 3)

    print(f"TTFB: {ttfb}ms, generation: {generation}s, duration: {duration}s, RTX: {round(duration / generation, 2)}")
    if failed_segments:
        print(f"Segments replaced with silence fallback: {failed_segments}")
    print(f"Saved streaming audio to 'streaming.wav' (sampling rate: {out_sr} Hz).")

    # Or use the following to display the audio in the jupyter notebook:
    # from IPython.display import Audio
    # display(Audio(data=audio, rate=out_sr))


if __name__ == "__main__":
    main()
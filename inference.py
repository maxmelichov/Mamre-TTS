#!/usr/bin/env python
"""
Long Text Inference Script for Mamre-v1 with automatic silence trimming and
re-encoding for tail-frame continuity.
"""

import torch
import torchaudio
import torch.nn.functional as F
import re
import os
import gc
from pydub import AudioSegment
from tqdm import tqdm
from typing import List
from Mamre.model import Mamre
from Mamre.conditioning import make_cond_dict, phonemize
from Mamre.codebook_pattern import apply_delay_pattern

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_text_into_segments(text: str) -> List[str]:
    """Split text into segments by newline characters."""
    segments = re.split(r'[\n]', text)
    return [seg.strip() for seg in segments if seg.strip()]

def trim_silence(wav: torch.Tensor,
                 sample_rate: int,
                 thresh: float = 0.1,
                 min_silence_len: float = 0.05,
                 keep_silence: float = 0.02) -> torch.Tensor:
    """
    Trim leading & trailing silence from a mono waveform,
    but only if those silent regions are at least min_silence_len long.
    Optionally, keep a small portion of the silence at each end.

    wav: Tensor of shape [1, T]
    thresh: amplitude threshold below which we consider “silence”
    min_silence_len: minimum duration (in seconds) of a silent region to trim
    keep_silence: duration (in seconds) of silence to keep at each end
    """
    min_silence_samples = int(min_silence_len * sample_rate)
    keep_silence_samples = int(keep_silence * sample_rate)

    win_size = max(min_silence_samples, 1)
    abs_wav = wav.abs()
    kernel = torch.ones(1, 1, win_size, device=wav.device) / win_size
    energy = F.conv1d(abs_wav.unsqueeze(0), kernel, padding=win_size//2).squeeze(0)

    mask = (energy > thresh)[0]
    non_silent = mask.nonzero(as_tuple=False).squeeze()

    if non_silent.numel() == 0:
        return wav

    first_ns = non_silent[0].item()
    last_ns  = non_silent[-1].item()
    total_len = wav.shape[1]

    leading_silence = first_ns
    trailing_silence = total_len - 1 - last_ns

    # Only trim if silence is long enough, but keep a bit of it
    start = max(0, first_ns - keep_silence_samples) if leading_silence >= min_silence_samples else 0
    end   = min(total_len, last_ns + 1 + keep_silence_samples) if trailing_silence >= min_silence_samples else total_len

    return wav[:, start:end]


def main():
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Prepare directories
    os.makedirs("generated_segments", exist_ok=True)

    # Load model
    print("Loading model from notmax123/MamreTTS...")
    model = Mamre.from_pretrained("notmax123/MamreTTS", model_filename="MamreV1.safetensors", device="cpu")

    # Optional checkpoint
    ckpt = "checkpoints/train_epoch_1.pt"
    if os.path.isfile(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print("Checkpoint loaded.")
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")

    model.eval()
    model.to(device)
    model.autoencoder.dac.eval()
    model.autoencoder.dac.to(device)

    # Speaker embedding
    audio_path = "voices/male1.wav"
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav, sr, model.autoencoder.sampling_rate)
    sr = model.autoencoder.sampling_rate
    print("Computing speaker embedding...")
    
    with torch.no_grad():
        speaker = model.make_speaker_embedding(wav, sr)

    # Text to synthesize
    long_text =  """
        jeʁuʃalˈajim hˈi ʔˈiʁ ʔatikˈa vaχaʃuvˈa bimjuχˈad, ʃemeχilˈa betoχˈa ʃχavˈot ʁabˈot ʃˈel histˈoʁja,
        taʁbˈut veʁuχanijˈut ʃenimʃaχˈot ʔalfˈej ʃanˈim, vehˈi mehavˈa mokˈed meʁkazˈi liʃlˈoʃet
        hadatˈot haɡdolˈot, jahadˈut, natsʁˈut veʔislˈam.
        ʃemitχabʁˈot jˈaχad bemakˈom ʔeχˈad jiχudˈi, malˈe ʔenˈeʁɡija umuʁkavˈut,
        ʃˈam ʔefʃˈaʁ limtsˈo ʔataʁˈim kdoʃˈim, ʃχunˈot ʔatikˈot, veʃvakˈim tsivʔonijˈim,
        vekˈol pinˈa mesapˈeʁet sipˈuʁ ʃˈel tkufˈot ʃonˈot, ʔanaʃˈim ʃonˈim veʔejʁuʔˈim ʃehiʃpˈiʔu ʔˈal hahistˈoʁja ʃˈel haʔolˈam kulˈo,
        mˈa ʃehofˈeχ ʔet jeʁuʃalˈajim lˈo ʁˈak leʔˈiʁ ɡeʔoɡʁˈafit ʔˈela ɡˈam lemeʁkˈaz ʃˈel zehˈut,
        ʔemunˈa vezikaʁˈon kolektˈivi ʃemamʃˈiχ leʔoʁˈeʁ haʃʁaʔˈa ulχabˈeʁ bˈen ʔanaʃˈim meʁˈeka ʃonˈe mikˈol kitsvˈot tevˈel.
         """
    segments = split_text_into_segments(long_text)
    print(f"Split text into {len(segments)} segments")

    all_generated = []
    prev_tail_codes = None
    TAIL_FRAMES = 8

    for i, segment in enumerate(tqdm(segments, desc="Generating")):
        print(f"\nSegment {i+1}/{len(segments)}: {segment[:40]}")
        attempt = 0
        min_len = 20 if int(len(segment) * 2) < 150 else 100
        min_length = max(min_len, int(len(segment) * 3.5))
        max_new_tokens = max(200, int(len(segment) * 6.5))
        while True:
            attempt += 1
            # Prepare conditioning
            cond = make_cond_dict(text=segment,
                                  language="en-us",
                                  speaking_rate=9.0,
                                  speaker=speaker)
            conditioning = model.prepare_conditioning(cond)

            # Generate codes (with or without prefix)
            with torch.no_grad():
                if prev_tail_codes is not None:
                    codes = model.generate(
                        conditioning,
                        audio_prefix_codes=prev_tail_codes,
                        max_new_tokens=max_new_tokens
                    )
                else:
                    codes = model.generate(
                        conditioning,
                        max_new_tokens=max_new_tokens
                    )

            # # Decode to waveform
            # with torch.no_grad():
            #     wav_out = model.autoencoder.decode(codes).cpu()[0]

            # # Trim long silences
            # trimmed = trim_silence(wav_out, sr,
            #                         thresh=0.05,
            #                         min_silence_len=0.1,
            #                         keep_silence=0.02)

            
            if codes.shape[-1] >= min_length and codes.shape[-1] <= max_new_tokens:
                break
            # If too short, try again
            print(f"Segment {i+1}: generated length {codes.shape[-1]} < {min_length}, retrying (attempt {attempt})...")

        # Re-encode trimmed for next-tail continuity
        with torch.no_grad():
            # inp = codes.unsqueeze(0).to(device)  # [1,1,T]
            wav_out = model.autoencoder.decode(codes).cpu()[0]

        if TAIL_FRAMES:
            prev_tail_codes = codes[..., -TAIL_FRAMES:]
        
        if i > 0:  # Skip first segment as it has no prefix
            # Calculate how many audio samples correspond to TAIL_FRAMES
            samples_per_frame = wav_out.shape[-1] // codes.shape[-1]
            tail_samples = TAIL_FRAMES * samples_per_frame
            wav_out = wav_out[..., tail_samples:]

        # Save trimmed segment
        seg_path = f"generated_segments/segment_{i:03d}.wav"
        torchaudio.save(seg_path, wav_out, sr)
        all_generated.append(seg_path)


    pattern = re.compile(r'([^.,]+)([.,]?)')
    segments_with_punct = [ (m.group(1).strip(), m.group(2)) 
                            for m in pattern.finditer(long_text) 
                            if m.group(1).strip() ]

    # 2) Build a silence map (in ms) from the punctuation
    silence_map = []
    for text_seg, punct in segments_with_punct:
        if punct == ',':
            silence_map.append(50)
        elif punct == '.':
            silence_map.append(50)
        else:
            silence_map.append(0)

    # 3) Combine the generated wavs using that map
    crossfade_ms = 20

    combined = AudioSegment.empty()
    for idx, file_path in enumerate(all_generated):
        seg_audio = AudioSegment.from_file(file_path)
        seg_dur   = len(seg_audio)  # duration in ms

        if idx == 0:
            combined = seg_audio
        else:
            # only use crossfade if the segment is longer than crossfade_ms
            if seg_dur >= crossfade_ms:
                combined = combined.append(seg_audio, crossfade=crossfade_ms)
            else:
                # too short for crossfade → just concatenate
                combined = combined + seg_audio

        # add the punctuation‑based silence (skip after last segment)
        if idx < len(silence_map) - 1 and silence_map[idx] > 0:
            combined += AudioSegment.silent(duration=silence_map[idx])

    # 4) Export
    out_path = "male1_hebrew.wav"
    combined.export(out_path, format="wav")
    print(f"Saved combined audio to {out_path}")

if __name__ == "__main__":
    main()

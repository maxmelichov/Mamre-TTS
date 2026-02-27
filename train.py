#!/usr/bin/env python
"""Train Zonos/Mamre TTS. CSV columns: filename, phonemes; optional quality."""

import argparse
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from Mamre.autoencoder import DACAutoencoder
from Mamre.model import Mamre
from Mamre.conditioning import make_cond_dict, make_cond_dict_train
from Mamre.codebook_pattern import apply_delay_pattern, revert_delay_pattern


def shift_right(codes: torch.Tensor, mask_token: int) -> torch.Tensor:
    B, N, T = codes.shape
    shifted = torch.full((B, N, T), mask_token, dtype=codes.dtype, device=codes.device)
    shifted[:, :, 1:] = codes[:, :, :-1]
    return shifted


def revert_delay_pattern_logits(logits: torch.Tensor) -> torch.Tensor:
    B, n_q, L, V = logits.shape
    # For each codebook head, select indices from k+1 to (L - n_q + k + 1)
    reverted = torch.stack(
        [logits[:, k, k+1 : L - n_q + k + 1, :] for k in range(n_q)],
        dim=1
    )
    return reverted


class AudioDataset(Dataset):
    def __init__(self, csv_file, target_sample_rate):
        self.data = pd.read_csv(csv_file)
        if self.data.empty:
            raise ValueError(f"No data found in {csv_file}")
        self.target_sample_rate = target_sample_rate
        self.has_quality_column = 'quality' in self.data.columns
        before = len(self.data)
        self.data = self.data[self.data['filename'].apply(os.path.isfile)].reset_index(drop=True)
        skipped = before - len(self.data)
        if skipped:
            print(f"[Dataset] Skipped {skipped} rows with missing audio files ({len(self.data)} remaining).")
        if self.data.empty:
            raise ValueError(f"No valid audio files found in {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        wav_path = row['filename']
        # print(f"Loading {wav_path}")
        text = str(row['phonemes'])
        waveform, sr = torchaudio.load(wav_path)  # waveform: (channels, time)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
            sr = self.target_sample_rate
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Use quality from CSV if available, otherwise use path-based heuristics
        if self.has_quality_column:
            quality = float(row['quality'])
        else:
            if "saspeech_manual" in wav_path:
                quality = 1.0
            elif "motekspeech_v4" in wav_path or "ranspeech_v8" in wav_path:
                quality = 0.75
            elif "Part2" in wav_path:
                quality = 0.85
            # it ok for train only on bad data
            elif "saspeech_automatic" in wav_path:
                quality = 0.75
            else: # openai data
                quality = 1.0

        return waveform, sr, text, quality


def collate_fn(batch):
    waveforms, srs, texts, qualities = zip(*batch)
    unpadded_waveforms = list(waveforms)
    lengths = torch.tensor([w.shape[-1] for w in waveforms])
    waveforms_for_pad = [w.squeeze(0).t() for w in waveforms]
    waveforms_for_pad = [w.unsqueeze(1) for w in waveforms_for_pad]
    padded = nn.utils.rnn.pad_sequence(waveforms_for_pad, batch_first=True)
    padded = padded.transpose(1, 2)
    sr = srs[0]
    try:
        quality_tensor = torch.tensor(qualities, dtype=torch.float32)
    except Exception:
        quality_tensor = None
    return padded, sr, texts, quality_tensor, unpadded_waveforms, lengths


def _state_dict_for_save(model):
    """Return state_dict without 'module.' prefix so checkpoints load on single GPU."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_state_dict(model, state_dict):
    """Load state_dict, stripping 'module.' prefix if present."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.load_state_dict(state_dict, strict=False)


def train(args):
    cuda_available = torch.cuda.is_available()
    device0 = torch.device("cuda:0" if cuda_available else "cpu")
    device1 = torch.device("cuda:1" if (cuda_available and torch.cuda.device_count() > 1) else device0)
    n_gpus = torch.cuda.device_count() if cuda_available else 0
    print(f"Model device: {device0}; aux device: {device1}; GPUs: {n_gpus}")

    autoencoder = DACAutoencoder()
    autoencoder.dac.eval()
    autoencoder.dac.requires_grad_(False)
    autoencoder.dac.to(device1)

    print("Loading model from Hugging Face: Zyphra/Zonos-v0.1-hybrid")
    model = Mamre.from_pretrained("Zyphra/Zonos-v0.1-hybrid")
    checkpoint_path = getattr(args, "checkpoint", "") or ""
    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            _load_state_dict(model, state)
            print("Checkpoint loaded.")
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")
    model.to(device0)
    if device1 != device0:
        model.speaker_device = str(device1)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs; checkpoints saved without 'module.' prefix.")
    base = model.module if isinstance(model, nn.DataParallel) else model
    base.train()
    for param in base.prefix_conditioner.parameters():
        param.requires_grad = False
    if base.spk_clone_model is not None:
        for param in base.spk_clone_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")

    dataset = AudioDataset(args.csv_file, target_sample_rate=autoencoder.sampling_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps//2, eta_min=1e-6)

    for epoch in trange(args.epochs, desc="Epochs"):
        epoch_loss = 0.0
        processed_dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for i, (waveforms, sr, texts, quality, unpadded_waveforms, audio_lengths) in enumerate(processed_dataloader):
            try:
                waveforms = waveforms.to(device1)
                processed = [autoencoder.preprocess(w, sr) for w in waveforms]
                processed = nn.utils.rnn.pad_sequence(processed, batch_first=True).to(device1)

                with torch.no_grad():
                    latent_codes = autoencoder.encode(processed)
                latent_codes = latent_codes.long()
                latent_codes = latent_codes.to(device0)

                batch_size, n_codebooks, seq_len = latent_codes.shape
                eos_token = torch.full((batch_size, n_codebooks, 1),
                                         base.eos_token_id,
                                         device=device0,
                                         dtype=latent_codes.dtype)
                latent_codes = torch.cat([latent_codes, eos_token], dim=2)

                teacher_codes = shift_right(latent_codes, mask_token=base.masked_token_id)
                delayed_teacher_codes = apply_delay_pattern(teacher_codes, mask_token=base.masked_token_id)

                optimizer.zero_grad()
                hidden_states = base.embed_codes(delayed_teacher_codes)
                if batch_size == 1:
                    spk_emb = base.make_speaker_embedding(unpadded_waveforms[0].to(device1), sr).to(device0)
                    cond_dict = make_cond_dict_train(text=texts, speaker=spk_emb, language="he", device=device0)
                    prefix_cond = base.prefix_conditioner(cond_dict)
                else:
                    prefix_conds = []
                    for j, (unpadded_wav, text) in enumerate(zip(unpadded_waveforms, texts)):
                        spk_emb = base.make_speaker_embedding(unpadded_wav.to(device1), sr).to(device0)
                        cond_dict = make_cond_dict_train(text=[text], speaker=spk_emb, language="he", device=device0)
                        single_prefix_cond = base.prefix_conditioner(cond_dict)
                        prefix_conds.append(single_prefix_cond)
                    max_prefix_len = max(p.shape[1] for p in prefix_conds)
                    prefix_conds = [
                        torch.cat(
                            [p, torch.zeros(p.shape[0], max_prefix_len - p.shape[1], p.shape[2],
                                            device=p.device, dtype=p.dtype)],
                            dim=1
                        ) if p.shape[1] < max_prefix_len else p
                        for p in prefix_conds
                    ]
                    prefix_cond = torch.cat(prefix_conds, dim=0)
                
                full_hidden = torch.cat([prefix_cond, hidden_states], dim=1)
                backbone_out = base.backbone(full_hidden)
                pred_hidden = backbone_out[:, prefix_cond.shape[1]:, :]
                pred_logits = base.apply_heads(pred_hidden)

                pred_logits = revert_delay_pattern_logits(pred_logits)

                hop_length = 512
                latent_lengths = (audio_lengths.float() / hop_length).ceil().long().to(device0)
                latent_lengths = (latent_lengths + 1).clamp(max=latent_codes.shape[2])
                max_len = latent_codes.shape[2]
                mask = torch.arange(max_len, device=device0).unsqueeze(0) < latent_lengths.unsqueeze(1)

                loss_sum = 0.0
                for cb in range(n_codebooks):
                    loss_cb = criterion(
                        pred_logits[:, cb, :, :].reshape(-1, pred_logits.shape[-1]),
                        latent_codes[:, cb, :].reshape(-1)
                    ).view(batch_size, -1)
                    loss_cb = loss_cb * mask.float()
                    if quality is not None:
                        loss_cb = loss_cb * quality.unsqueeze(1).to(device0)
                    valid_tokens = mask.sum().clamp(min=1)
                    loss_sum += loss_cb.sum() / valid_tokens
                loss = loss_sum / n_codebooks

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (i + 1)
                processed_dataloader.set_postfix({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
                
            except torch.cuda.OutOfMemoryError:
                print(f"\nCUDA OOM on batch {i}. Skipping batch...")
                optimizer.zero_grad()
                if cuda_available:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                continue

            if cuda_available:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete – Average Loss: {avg_loss:.4f}")

        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"train_epoch_{epoch+1}.pt")
        torch.save(_state_dict_for_save(model), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        print("Generating sample audio...")
        base.eval()
        with torch.no_grad():
            test_wav_path = "voices/Female1.mp3"
            if os.path.isfile(test_wav_path):
                sample_wav, sample_sr = torchaudio.load(test_wav_path)
                if sample_wav.shape[0] > 1:
                    sample_wav = sample_wav.mean(dim=0, keepdim=True)
                spk_emb = base.make_speaker_embedding(sample_wav.to(device1), sample_sr)
            else:
                spk_emb = None

            sample_text = "jeʁuʃalˈajim hˈi ʔˈiʁ ʔatikˈa vaχaʃuvˈa bimjuχˈad, ʃemeχilˈa betoχˈa ʃχavˈot ʁabˈot ʃˈel histˈoʁja,"
            sample_cond_dict = make_cond_dict(text=sample_text, speaker=spk_emb, language="he", device=device0, speaking_rate=9.0)
            sample_conditioning = base.prepare_conditioning(sample_cond_dict)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_codes = base.generate(sample_conditioning, batch_size=1, cfg_scale=4.0)
            out_wav = base.autoencoder.decode(out_codes.to(device1)).cpu()
            sample_out_path = os.path.join(args.save_dir, f"sample_epoch_{epoch+1}.wav")
            torchaudio.save(sample_out_path, out_wav[0], base.autoencoder.sampling_rate)
            print(f"Sample saved at {sample_out_path}")
            base.train()

    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Zonos/Mamre TTS")
    parser.add_argument("--csv_file", type=str, default="data/best_menual_data.csv",
                        help="CSV with 'filename' and 'phonemes' (optional 'quality').")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints and sample audio.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (set to 1 to avoid padding).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval (iterations).")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of dataloader workers.")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint to resume from (optional).")
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()

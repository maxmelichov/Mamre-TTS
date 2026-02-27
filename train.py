#!/usr/bin/env python
"""
Improved Training Script for Zonos-v0.1 with tqdm Progress Bars

This script trains the autoregressive Zonos model to predict sequences of audio tokens.
It incorporates many training insights, including:
  - Two-phase training: "pretrain" vs. "finetune"
  - Conditioning on normalized text and optional speaker/prosody inputs
  - Use of the high-bitrate DAC autoencoder producing 774 tokens/sec (across 9 codebooks)
  - Sample audio generation at the end of each epoch for progress monitoring

Data is loaded from a CSV file expected to have at least:
   • "file_name": path (without extension) to the audio file (WAV)
   • "english": the transcription for conditioning
Optionally, a "quality" column can be provided.

Usage example:
    python train_improved.py --csv_file data.csv --phase finetune --epochs 20 --batch_size 8 --learning_rate 1e-4 --save_dir ./checkpoints
"""

import argparse
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd  # For reading CSV files

# Import model components from your Zonos code
from Mamre.autoencoder import DACAutoencoder
from Mamre.model import Mamre
from Mamre.conditioning import make_cond_dict, make_cond_dict_train
from Mamre.codebook_pattern import apply_delay_pattern, revert_delay_pattern
# -----------------------------------------------------------------------------
# Helper Function: Shift Right
# -----------------------------------------------------------------------------
def shift_right(codes: torch.Tensor, mask_token: int) -> torch.Tensor:
    """
    Shift the ground truth audio tokens to the right by one time-step,
    filling the first token in each codebook with the mask token.
    
    Args:
        codes: Tensor of shape [B, num_codebooks, T] (ground truth codes)
        mask_token: The token id used for masking (e.g. model.masked_token_id)
    Returns:
        shifted codes: Tensor of shape [B, num_codebooks, T]
    """
    B, N, T = codes.shape
    shifted = torch.full((B, N, T), mask_token, dtype=codes.dtype, device=codes.device)
    shifted[:, :, 1:] = codes[:, :, :-1]
    return shifted

# -----------------------------------------------------------------------------
# Helper Function: Revert Delay Pattern for Logits
# -----------------------------------------------------------------------------
def revert_delay_pattern_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Given logits produced from delayed teacher forcing inputs, slice the
    time dimension appropriately to revert the delay pattern so that the
    output matches the target latent token sequence length.

    Args:
        logits: Tensor of shape [B, n_codebooks, L, vocab_size] where
                L = T + n_codebooks (T is original target length).
    Returns:
        Tensor of shape [B, n_codebooks, T, vocab_size]
    """
    B, n_q, L, V = logits.shape
    # For each codebook head, select indices from k+1 to (L - n_q + k + 1)
    reverted = torch.stack(
        [logits[:, k, k+1 : L - n_q + k + 1, :] for k in range(n_q)],
        dim=1
    )
    return reverted

# -----------------------------------------------------------------------------
# Dataset Definition (using CSV input)
# -----------------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, csv_file, target_sample_rate):
        """
        Args:
            csv_file (str): Path to a CSV file with columns "filename" and "phonemes".
                            Optionally, a "quality" column can be included.
            target_sample_rate (int): The sample rate expected by the autoencoder/model.
        """
        self.data = pd.read_csv(csv_file)
        if self.data.empty:
            raise ValueError(f"No data found in {csv_file}")
        self.target_sample_rate = target_sample_rate
        self.has_quality_column = 'quality' in self.data.columns
        # Drop rows whose audio file does not exist on disk
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
    
# =============================================================================
#  Collate Function (handles variable-length audio and quality if provided)
# =============================================================================

def collate_fn(batch):
    """
    Collates a batch of items.
    Each item: (waveform, sr, text, quality)
    Returns:
       - padded_waveforms: (batch, 1, time)
       - sr: sample rate (assumed the same for all)
       - texts: list of texts
       - quality: tensor of quality scores (or None)
       - unpadded_waveforms: list of unpadded (1, time) tensors for clean speaker embedding
       - lengths: tensor of original waveform lengths (in samples) for loss masking
    """
    waveforms, srs, texts, qualities = zip(*batch)
    # Preserve original (unpadded) waveforms for speaker embedding extraction
    unpadded_waveforms = list(waveforms)  # each is (1, time) at target_sample_rate
    lengths = torch.tensor([w.shape[-1] for w in waveforms])
    # Convert each waveform (1, time) -> (time, 1), then pad, then transpose back to (batch, 1, time)
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


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train(args):
    # Configure devices
    cuda_available = torch.cuda.is_available()
    device0 = torch.device("cuda:0" if cuda_available else "cpu")
    device1 = torch.device("cuda:1" if (cuda_available and torch.cuda.device_count() > 1) else device0)
    print(f"Model device: {device0}; Aux device: {device1}")

    # -------------------------------------------------------------------------
    # Load and prepare the DAC autoencoder (frozen for training)
    # -------------------------------------------------------------------------
    autoencoder = DACAutoencoder()
    autoencoder.dac.eval()
    autoencoder.dac.requires_grad_(False)
    # Place autoencoder on auxiliary device to free memory on main device
    autoencoder.dac.to(device1)

    # -------------------------------------------------------------------------
    # Load the Zonos model from Hugging Face Hub.
    # Repository: Zyphra/Zonos-v0.1-hybrid
    # -------------------------------------------------------------------------
    print("Loading model from Hugging Face: Zyphra/Zonos-v0.1-hybrid")
    model = Mamre.from_pretrained("Zyphra/Zonos-v0.1-hybrid")
    # Load the checkpoint (replace with your checkpoint path)
    checkpoint_path = ""#"bad_data.pt"#"checkpoints/zonos_epoch_4.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded.")
    except Exception as e: 
        print(f"Warning: could not load checkpoint: {e}")
    model.to(device0)
    # If we have two devices, offload speaker embedder to device1
    if device1 != device0:
        model.speaker_device = str(device1)
    model.train()  # Set to training mode
    for param in model.prefix_conditioner.parameters():
        param.requires_grad = False

    # for param in model.embeddings.parameters():
    #     param.requires_grad = False

    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    if model.spk_clone_model is not None:
        for param in model.spk_clone_model.parameters():
            param.requires_grad = False
    
    # for param in model.heads.parameters():
    #     param.requires_grad = False
    

        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    # -------------------------------------------------------------------------
    # Create the training dataset and dataloader.
    # -------------------------------------------------------------------------
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
    
    # -----------------------------------------------------------------------------
    # Main Training Loop
    # -----------------------------------------------------------------------------
    for epoch in trange(args.epochs, desc="Epochs"):
        epoch_loss = 0.0
        processed_dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for i, (waveforms, sr, texts, quality, unpadded_waveforms, audio_lengths) in enumerate(processed_dataloader):
            try:
                waveforms = waveforms.to(device1)

                # Preprocess: resample & pad using the autoencoder helper.
                processed = [autoencoder.preprocess(w, sr) for w in waveforms]
                processed = nn.utils.rnn.pad_sequence(processed, batch_first=True).to(device1)

                # Encode audio into latent tokens (with no gradient).
                with torch.no_grad():
                    latent_codes = autoencoder.encode(processed)
                latent_codes = latent_codes.long()  # shape: [B, num_codebooks, T_raw]
                # Move codes to model device for training
                latent_codes = latent_codes.to(device0)

                # -------------------------- Teacher Forcing Preparation with Delay Pattern --------------------------
                # Append EOS token to the latent codes.
                batch_size, n_codebooks, seq_len = latent_codes.shape
                eos_token = torch.full((batch_size, n_codebooks, 1),
                                         model.eos_token_id,
                                         device=device0,
                                         dtype=latent_codes.dtype)
                latent_codes = torch.cat([latent_codes, eos_token], dim=2)  # New T = T_raw + 1

                # Create teacher forcing inputs by shifting right...
                teacher_codes = shift_right(latent_codes, mask_token=model.masked_token_id)
                # ... and then applying the delay pattern.
                delayed_teacher_codes = apply_delay_pattern(teacher_codes, mask_token=model.masked_token_id)
                # delayed_teacher_codes shape: [B, n_codebooks, T + n_codebooks]
                # -------------------------- End Teacher Forcing --------------------------

                optimizer.zero_grad()
                
                # Forward pass: embed delayed teacher tokens and build conditioning prefix.
                hidden_states = model.embed_codes(delayed_teacher_codes)

                # Use *unpadded* waveforms for speaker embedding so the cloning model
                # never sees batch-padding zeros, which corrupt the speaker latent space.
                if batch_size == 1:
                    spk_emb = model.make_speaker_embedding(unpadded_waveforms[0].to(device1), sr).to(device0)
                    cond_dict = make_cond_dict_train(text=texts, speaker=spk_emb, language="he", device=device0)
                    prefix_cond = model.prefix_conditioner(cond_dict)
                else:
                    prefix_conds = []
                    for j, (unpadded_wav, text) in enumerate(zip(unpadded_waveforms, texts)):
                        spk_emb = model.make_speaker_embedding(unpadded_wav.to(device1), sr).to(device0)
                        cond_dict = make_cond_dict_train(text=[text], speaker=spk_emb, language="he", device=device0)
                        single_prefix_cond = model.prefix_conditioner(cond_dict)
                        prefix_conds.append(single_prefix_cond)
                    # Different texts produce different-length prefix tensors; pad to the
                    # longest one along dim=1 before stacking along the batch dim.
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
                backbone_out = model.backbone(full_hidden)
                # Discard the prefix part.
                pred_hidden = backbone_out[:, prefix_cond.shape[1]:, :]
                pred_logits = model.apply_heads(pred_hidden)
                # pred_logits currently has shape [B, n_codebooks, (T + n_codebooks), vocab_size]

                # Revert the delay pattern on the logits to align with target tokens.
                pred_logits = revert_delay_pattern_logits(pred_logits)
                # Now pred_logits shape: [B, n_codebooks, T, vocab_size]

                # -------------------- Compute Loss --------------------
                # DAC hop length is 512 samples at 44100 Hz → latent rate ≈ 86 Hz.
                # Build a mask so padding tokens (from batch collation) contribute 0 loss.
                hop_length = 512  # DAC encoder stride
                latent_lengths = (audio_lengths.float() / hop_length).ceil().long().to(device0)
                # +1 for the EOS token appended after encoding
                latent_lengths = (latent_lengths + 1).clamp(max=latent_codes.shape[2])
                max_len = latent_codes.shape[2]
                # mask shape: [B, T]  — True where the token is real (not padding)
                mask = torch.arange(max_len, device=device0).unsqueeze(0) < latent_lengths.unsqueeze(1)

                loss_sum = 0.0
                for cb in range(n_codebooks):
                    loss_cb = criterion(
                        pred_logits[:, cb, :, :].reshape(-1, pred_logits.shape[-1]),
                        latent_codes[:, cb, :].reshape(-1)
                    ).view(batch_size, -1)
                    # Zero out loss on padding positions
                    loss_cb = loss_cb * mask.float()
                    if quality is not None:
                        loss_cb = loss_cb * quality.unsqueeze(1).to(device0)
                    # Average only over valid (non-padding) tokens
                    valid_tokens = mask.sum().clamp(min=1)
                    loss_sum += loss_cb.sum() / valid_tokens
                loss = loss_sum / n_codebooks
                # -------------------- End Compute Loss --------------------

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (i + 1)
                processed_dataloader.set_postfix({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
                
            except torch.cuda.OutOfMemoryError:
                # Handle CUDA OOM error
                print(f"\nCUDA out of memory error on batch {i}. Cleaning memory and continuing...")
                
                # Skip this batch
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

        # Save checkpoint after each epoch.
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"train_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # --------------------- Dynamic Sample Generation ---------------------
        print("Generating sample audio for debug/inference...")
        model.eval()  # Set to eval mode for generation
        with torch.no_grad():
            # Use a test wav file for speaker embedding.
            test_wav_path = "excerpt_7-04_to_7-14.wav"
            if os.path.isfile(test_wav_path):
                sample_wav, sample_sr = torchaudio.load(test_wav_path)
                if sample_wav.shape[0] > 1:
                    sample_wav = sample_wav.mean(dim=0, keepdim=True)
                spk_emb = model.make_speaker_embedding(sample_wav.to(device1), sample_sr)
            else:
                spk_emb = None  # Fallback

            # sample_text = """z'ehh k'ol shenotsar 'al yd'ey biynahh mlahoot'iyt bk'arov an'iy af'arsem od' prat'iym sheyihhyehh l'ahem shavoo'a mad'hhiym"""
            sample_text = "jeʁuʃalˈajim hˈi ʔˈiʁ ʔatikˈa vaχaʃuvˈa bimjuχˈad, ʃemeχilˈa betoχˈa ʃχavˈot ʁabˈot ʃˈel histˈoʁja,"
            sample_cond_dict = make_cond_dict(text=sample_text, speaker=spk_emb, language="he", device=device0, speaking_rate=9.0)
            sample_conditioning = model.prepare_conditioning(sample_cond_dict)

            # Optionally use autocast for faster generation.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_codes = model.generate(sample_conditioning, batch_size=1, cfg_scale=4.0)
            out_wav = model.autoencoder.decode(out_codes.to(device1)).cpu()
            sample_out_path = os.path.join(args.save_dir, f"sample_epoch_{epoch+1}.wav")
            torchaudio.save(sample_out_path, out_wav[0], model.autoencoder.sampling_rate)
            print(f"Sample saved at {sample_out_path}")
            model.train()  # Reset to training mode
        # ------------------------------------------------------------------------------
        
    print("Training complete.")

# -----------------------------------------------------------------------------
# Argument Parsing and Main Function
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Zonos-v0.1 model")
    parser.add_argument("--csv_file", type=str, default="data/best_menual_data.csv",
                        help="Path to CSV file containing 'file_name' and 'english' columns (optionally 'quality').")
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
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()

"""Small audio IO / preprocessing helpers shared across the data pipeline and training code."""
import random

import torch
import torchaudio


def load_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load an audio file, downmix to mono, resample to target_sample_rate.

    Returns a 1-D float32 tensor in [-1, 1].
    """
    wav, sr = torchaudio.load(path)  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav.squeeze(0)


def random_crop_or_pad(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Return exactly `num_samples` samples: random crop if longer, zero-pad if shorter."""
    length = wav.shape[-1]
    if length == num_samples:
        return wav
    if length > num_samples:
        start = random.randint(0, length - num_samples)
        return wav[start:start + num_samples]
    pad = num_samples - length
    return torch.nn.functional.pad(wav, (0, pad))


def normalize_peak(wav: torch.Tensor, target_db: float = -1.0) -> torch.Tensor:
    """Peak-normalize a waveform to `target_db` dBFS to reduce loudness variance across sources."""
    peak = wav.abs().max().clamp(min=1e-8)
    target_amp = 10 ** (target_db / 20)
    return wav * (target_amp / peak)

"""
PyTorch Dataset that reads a manifest.jsonl (see prepare_deam.py / prepare_pmemo.py / auto_label.py)
and yields raw (waveform, caption, valence, arousal) tuples. EnCodec tokenization is done inside
the training loop (via the MusicGen processor) rather than here, so this stays model-agnostic and
easy to inspect/debug.
"""
import json
import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from utils.audio_utils import load_audio, normalize_peak, random_crop_or_pad


class EmotionMusicDataset(Dataset):
    def __init__(
        self,
        manifest_paths: List[str],
        sample_rate: int = 32000,
        clip_seconds: float = 10.0,
        extra_pseudo_labeled_manifest: Optional[str] = None,
        pseudo_label_weight: float = 1.0,
    ):
        """
        Args:
            manifest_paths: one or more .jsonl manifests with ground-truth V-A labels.
            sample_rate: must match the MusicGen/EnCodec backbone's expected sample rate (32kHz
                for facebook/musicgen-small/medium).
            clip_seconds: length of the audio segment fed to the model per training step.
            extra_pseudo_labeled_manifest: optional path to an auto-labeled manifest
                (data/auto_label.py output) to mix in for Section 4.3's "expand the training set".
            pseudo_label_weight: probability of *keeping* a pseudo-labeled example when
                __getitem__ builds its index (use < 1.0 to down-weight noisier pseudo labels
                relative to ground truth without changing your DataLoader's sampler).
        """
        self.sample_rate = sample_rate
        self.num_samples = int(clip_seconds * sample_rate)

        self.records = []
        for path in manifest_paths:
            self.records.extend(self._load_manifest(path))

        if extra_pseudo_labeled_manifest and os.path.exists(extra_pseudo_labeled_manifest):
            pseudo_records = self._load_manifest(extra_pseudo_labeled_manifest)
            if pseudo_label_weight < 1.0:
                keep = int(len(pseudo_records) * pseudo_label_weight)
                pseudo_records = pseudo_records[:keep]
            self.records.extend(pseudo_records)

        if not self.records:
            raise ValueError("EmotionMusicDataset: no records loaded — check your manifest paths.")

    @staticmethod
    def _load_manifest(path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        wav = load_audio(rec["audio_path"], self.sample_rate)
        wav = random_crop_or_pad(wav, self.num_samples)
        wav = normalize_peak(wav)
        return {
            "waveform": wav,                       # (num_samples,)
            "caption": rec["caption"],
            "valence": float(rec["valence"]),
            "arousal": float(rec["arousal"]),
            "is_pseudo_label": bool(rec.get("is_pseudo_label", False)),
        }


def collate_fn(batch):
    waveforms = torch.stack([b["waveform"] for b in batch])          # (B, num_samples)
    captions = [b["caption"] for b in batch]
    valence = torch.tensor([b["valence"] for b in batch], dtype=torch.float32)
    arousal = torch.tensor([b["arousal"] for b in batch], dtype=torch.float32)
    is_pseudo = torch.tensor([b["is_pseudo_label"] for b in batch], dtype=torch.bool)
    return {
        "waveforms": waveforms,
        "captions": captions,
        "valence": valence,
        "arousal": arousal,
        "is_pseudo_label": is_pseudo,
    }

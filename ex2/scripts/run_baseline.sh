#!/usr/bin/env bash
# Phase 1 of the proposal's timeline: "reproduce a MusicGen inference baseline on AudioCraft".
# Here we reproduce it via the equivalent, easier-to-install HF `transformers` port of MusicGen
# (same pretrained weights family, `facebook/musicgen-small`) so the rest of the repo (which is
# built on `transformers`) can reuse this exact code path.
set -euo pipefail

python - <<'PY'
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading facebook/musicgen-small on {device} ...")

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

prompts = [
    "upbeat acoustic guitar, sunny afternoon, folk",
    "dark ambient drone, tense and unsettling",
]
inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)

sample_rate = model.config.audio_encoder.sampling_rate
audio_values = model.generate(**inputs, max_new_tokens=int(8 * model.config.audio_encoder.frame_rate))

for i, prompt in enumerate(prompts):
    wav = audio_values[i].squeeze().detach().cpu().numpy()
    out_path = f"baseline_{i}.wav"
    sf.write(out_path, wav, sample_rate)
    print(f"Wrote baseline sample for prompt '{prompt}' -> {out_path}")
PY

echo "Phase 1 baseline reproduction complete: see baseline_0.wav / baseline_1.wav"

"""
Export an untrained (identity) emotion adapter checkpoint for baseline evaluation.

The saved weights match a freshly initialized adapter (prefix_scale=0, etc.),
i.e. the same state as running generate.py without --checkpoint.

Usage:
    python scripts/export_identity_checkpoint.py
    python evaluate.py --checkpoint checkpoints/identity.pt --out_dir eval_outputs_baseline ...
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import torch
import yaml

from model.musicgen_emotion import MusicGenEmotion


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = MusicGenEmotion(
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        use_lora=cfg["model"]["use_lora"],
        lora_config=cfg["model"]["lora"],
        adapter_config=cfg["model"]["emotion_adapter"],
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    out_path = os.path.join(cfg["train"]["checkpoint_dir"], "identity.pt")
    torch.save(
        {"adapter": model.adapter.state_dict(), "config": cfg, "step": 0},
        out_path,
    )
    print(f"Saved identity adapter checkpoint to {out_path}")


if __name__ == "__main__":
    main()
